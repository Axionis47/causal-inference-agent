"""Adapt a compiled standalone plan to the retained numerical implementations.

The legacy plan exists only in memory. Its lineage fields are local calculation
handles, not claims that database artifacts or approvals have been committed.
Estimation, diagnostics and numerical support collection fail independently.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

import polars as pl
from pyfixest.estimation.feols_ import Feols

from causal.analysis.common import legacy_engine as engine
from causal.analysis.common.catalog import method_module
from causal.analysis.contracts import BoundaryError, CompiledPlan
from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import (
    EstimationPackV1,
    NuisanceProfileV1,
    load_estimation_packs,
)
from causal.analysis.methods.aipw import diagnostics as aipw_diagnostics
from causal.analysis.methods.aipw import estimation as aipw
from causal.analysis.methods.did import diagnostics as did_diagnostics
from causal.analysis.methods.did import estimation as did
from causal.analysis.methods.randomized import diagnostics as randomized_diagnostics
from causal.analysis.methods.randomized import estimation as randomized
from causal.analysis.methods.rdd import diagnostics as rdd_diagnostics
from causal.analysis.methods.rdd import estimation as rdd
from causal.shared.contracts import ArtifactRef, encode_contrast

Adapter = (randomized.RandomizedExperimentAdapter | aipw.ObservationalAipwAdapter
           | did.DifferenceInDifferencesAdapter | rdd.SharpRegressionDiscontinuityAdapter)
_METHODS: dict[str, tuple[str, str, type[Adapter]]] = {
    "randomized": ("randomized_experiment", "randomized-experiment-pack.v1",
                   randomized.RandomizedExperimentAdapter),
    "aipw": ("aipw", "aipw-pack.v1", aipw.ObservationalAipwAdapter),
    "did": ("did", "did-pack.v1", did.DifferenceInDifferencesAdapter),
    "rdd": ("sharp_rdd", "sharp-rdd-pack.v1", rdd.SharpRegressionDiscontinuityAdapter),
}


def _required(value: str | None) -> str:
    if value is None:
        raise BoundaryError("The compiled analysis contains an unresolved required setting.")
    return value


def _settings(plan: CompiledPlan) -> tuple[dict[str, str], ec.ValueMap, str]:
    config = plan.specification.configuration
    params: ec.ValueMap = {policy.id: policy.value
                          for policy in method_module(config.method).DEFINITION.fixed_policies
                          if not policy.id.startswith("nuisance.")}
    roles = {"treatment": _required(config.treatment_column),
             "unit_identifier": _required(config.unit_column),
             "outcome": plan.specification.design.outcome.column}
    if config.method == "randomized":
        optional = {"precision_covariate": config.precision_covariate_column,
                    "stratum": config.stratum_column, "cluster": config.cluster_column}
        params["specification"] = _required(config.estimator)
    elif config.method == "aipw":
        optional = {f"adjustment_covariate__{i}": name
                    for i, name in enumerate(config.covariate_columns)}
        params.update({"estimand": config.estimand, "propensity_bound": config.propensity_bound,
                       "nuisance_profile_id": config.nuisance_profile, "fold_count": config.fold_count})
    elif config.method == "did":
        optional = {"group": config.group_column, "cluster": config.cluster_column,
                    "adoption_time": config.adoption_column}
        roles["time"] = _required(config.time_column)
        params.update({"adoption_profile_id": _required(config.profile),
                  "adoption_time": config.adoption_column or config.adoption_time,
                  "comparison_cohort": config.comparison_cohort,
                  "anticipation_periods": config.anticipation_periods,
                  "reference_period": config.reference_period,
                  "event_window_lags": config.event_window_lags})
    else:
        optional = {"predetermined_covariate": config.covariate_column,
                    "cluster": config.cluster_column}
        roles["running_variable"] = _required(config.running_column)
        params.update({"cutoff": config.cutoff, "assignment_direction": config.assignment_direction,
                  "treated_value": _required(config.treated_value),
                  "comparator_value": _required(config.comparator_value),
                  "polynomial_order": config.polynomial_order, "kernel": config.kernel,
                  "bandwidth_selector": config.bandwidth_selector})
    roles.update({role: name for role, name in optional.items() if name is not None})
    contrast = (encode_contrast("treated", "never_treated") if config.method == "did"
                else encode_contrast(_required(config.treated_value),
                                     _required(config.comparator_value)))
    return roles, params, contrast


def prepare(plan: CompiledPlan, data: pl.DataFrame) -> ExecutionContext:
    """Build the numerical call using only a fully compiled, checked specification."""
    spec, config = plan.specification, plan.specification.configuration
    roles, params, contrast = _settings(plan)
    params["outcome_kind"] = spec.design.outcome.kind
    view = engine.estimator_view(data, roles)
    # Formula libraries and intermediate group counts see only generated identifiers.
    # The approved dataset and its human-facing column names remain unchanged.
    renamed = {name: f"analysis_input_{plan.plan_hash[:12]}_{i}"
               for i, name in enumerate(view.columns)}
    roles = {role: renamed[name] for role, name in roles.items()}
    if config.method == "did" and config.adoption_column is not None:
        params["adoption_time"] = renamed[config.adoption_column]
    method_id, pack_version, adapter_type = _METHODS[config.method]
    pack = load_estimation_packs().get(method_id, pack_version)
    if config.method == "aipw":
        # The new capability pins these learner settings rather than inheriting mutable
        # scientific choices from the legacy registry's primary-profile designation.
        nuisance_policy = {policy.id.removeprefix("nuisance."): policy.value
                           for policy in method_module("aipw").DEFINITION.fixed_policies
                           if policy.id.startswith("nuisance.")}
        nuisance = NuisanceProfileV1(
            profile_id=config.nuisance_profile, role="primary",
            propensity_learner=str(nuisance_policy.pop("propensity_learner")),
            outcome_learner=str(nuisance_policy.pop("outcome_learner")),
            hyperparameters=nuisance_policy)
        pack = pack.model_copy(update={"nuisance_profiles": (nuisance,)})
    # These handles satisfy retained numerical result types; they never leave this shim
    # as persisted lineage. The standalone evidence records the actual plan/data hashes.
    handle = ArtifactRef(artifact_id=f"in-memory:analysis:{plan.plan_hash}",
                         content_hash=plan.plan_hash)
    estimator_id, estimator_version = pack.estimator_id, pack.estimator_version
    if config.method == "did":
        profile = pack.estimator_profile(_required(config.profile))
        estimator_id, estimator_version = profile.estimator_id, profile.estimator_version
    folded = config.method == "aipw"
    legacy = ec.EstimationPlanV1(
        parents=(handle,), versions={"analysis_capability": plan.capability_version},
        context_manifest=handle, plan_revision=1,
        method_id=method_id, method_pack_version=pack_version, estimand_id=_required(config.estimand),
        population_id="fixed_design_population", timeframe_id="fixed_design_timeframe",
        comparator_id=contrast.partition("_vs_")[2], outcome_id=spec.design.outcome.column,
        unit_id=roles["unit_identifier"], role_columns=roles,
        row_set_hash=spec.dataset.content_hash, contrast_ids=(contrast,),
        required_sensitivity_ids=tuple(row.sensitivity_id for row in plan.sensitivities),
        figure_builder_ids=(), capacity_report=handle,
        numerical_tolerances={"sensitivity_magnitude": 0.25},
        estimator_id=estimator_id, estimator_version=estimator_version, seed=spec.seed,
        outcome_scale=spec.design.outcome.units,
        multiplicity_policy_id=str(params["multiplicity_policy_id"]) if config.method == "randomized" else None,
        primary_mask_rule_id=str(params["mask_rule_id"]),
        confidence_level=config.confidence_level, uncertainty_method=pack.uncertainty_method,
        finite_sample_correction=pack.finite_sample_correction, estimator_parameters=params,
        nuisance_profile_id=config.nuisance_profile if config.method == "aipw" else None,
        fold_count=config.fold_count if config.method == "aipw" else None,
        fold_assignment_rule_id=str(params["fold_assignment_rule_id"]) if folded else None,
        preprocessing_recipe_ids=("numeric_standardization",) if folded else (),
        required_diagnostics={row.diagnostic_id: cast(ec.DiagnosticSeverity, row.severity)
                              for row in plan.diagnostics if row.selected},
        numerical_failure_rule_ids=())
    return ExecutionContext(plan, adapter_type(handle), legacy, pack, view.rename(renamed),
                            {alias: original for original, alias in renamed.items()})


@dataclass(frozen=True)
class ExecutionContext:
    plan: CompiledPlan
    adapter: Adapter
    legacy_plan: ec.EstimationPlanV1
    pack: EstimationPackV1
    view: pl.DataFrame
    column_names: dict[str, str]

    def restore_names(self, text: str) -> str:
        pattern = "|".join(re.escape(name) for name in
                           sorted(self.column_names, key=len, reverse=True))
        return re.sub(f"(?:{pattern})(?![0-9])", lambda match: self.column_names[match.group()], text)

    def restore_measurements(self, harvest: Mapping[str, ec.ValueMap]) -> dict[str, ec.ValueMap]:
        return {name: {self.restore_names(key): self.restore_names(value)
                       if isinstance(value, str) else value for key, value in values.items()}
                for name, values in harvest.items()}

    def population_counts(self, item: ec.PrimaryContrastResultV1) -> dict[str, int]:
        if self.plan.specification.configuration.method != "rdd":
            return dict(item.contributing_counts)
        # The full frame selects bandwidths; N_h describes the local fit. Keep
        # those populations separately labelled, including for sensitivity fits.
        left = int(cast(int, item.method_quantities["effective_left"]))
        right = int(cast(int, item.method_quantities["effective_right"]))
        return {f"input_{key}": value for key, value in item.contributing_counts.items()} | {
            "local_fit_rows_left": left, "local_fit_rows_right": right}

    def fit(self, overrides: ec.ValueMap | None = None) -> engine.AdapterResult:
        return self.adapter.fit(self.view, self.legacy_plan, self.pack, overrides,
                                compute_diagnostics=False)

    def collect(self, result: engine.AdapterResult
                ) -> tuple[dict[str, ec.ValueMap], dict[str, str]]:
        """Collect checks from the primary fit; failures cannot erase its estimates."""
        harvested: dict[str, ec.ValueMap] = {}
        errors: dict[str, str] = {}
        wanted = {row.diagnostic_id for row in self.plan.diagnostics
                  if row.selected and row.applicability == "applicable"}
        plan, roles = self.legacy_plan, self.legacy_plan.role_columns
        params, method = plan.estimator_parameters, self.plan.specification.configuration.method

        def capture(names: set[str], compute: Callable[[], dict[str, ec.ValueMap]]) -> None:
            try:
                harvested.update(compute())
            except Exception as error:  # noqa: BLE001 -- retain every independent terminal result
                errors.update({name: self.restore_names(f"{type(error).__name__}: {error}") for name in names})

        if method == "randomized":
            capture(wanted, lambda: randomized_diagnostics.harvest(
                randomized._typed(self.view, roles), roles, plan, result.items,
                diagnostic_ids=frozenset(wanted)))
        elif method == "aipw":
            thresholds = next(({v.name: v.value for v in row.thresholds}
                               for row in self.plan.diagnostics
                               if row.diagnostic_id == "propensity_common_support"), {})
            capture(wanted, lambda: aipw_diagnostics.harvest(
                self.view, roles, plan, cast(aipw.CrossFitRun, result.fit), thresholds))
        elif method == "did":
            # Rebuild only the deterministic contribution view; do not fit again.
            frame = did._contributing(did._timing(self.view, roles, params), roles, params)
            cells, fitted = did._cells(frame, roles), cast(Feols | None, result.fit)
            if fitted is None:
                try:
                    fitted = did._event_fit(frame, roles)
                except Exception as error:  # noqa: BLE001 -- the simultaneous primary is independent
                    affected = {"event_study_pre_period_test", "reference_period_numerical_integrity"}
                    errors.update({name: self.restore_names(f"{type(error).__name__}: {error}")
                                   for name in wanted & affected})
            capture(wanted, lambda: did_diagnostics.harvest(
                frame, cells, roles, plan, params, fitted, include_figures=False))
            capture({"supporting_data"}, lambda: did.series(frame, cells, roles, plan, fitted))
        else:
            frame = self.view.drop_nulls([roles["running_variable"], roles["outcome"]])
            try:
                calls = rdd_diagnostics.measurement_calls(frame, roles, plan, params, result.fit)
            except Exception as error:  # noqa: BLE001 -- primary remains available
                errors.update({name: self.restore_names(f"{type(error).__name__}: {error}") for name in wanted})
                calls = {}
            for name, compute in calls.items():
                if name in wanted:
                    def single(key: str = name, operation: Callable[[], ec.ValueMap] = compute
                               ) -> dict[str, ec.ValueMap]:
                        return {key: operation()}
                    capture({name}, single)
            capture({"supporting_data"}, lambda: rdd.series(
                frame, roles, params, result.fit,
                harvested.get("sorting_and_other_policy_qualification", {})))
        return harvested, errors
