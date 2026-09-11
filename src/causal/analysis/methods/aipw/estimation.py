# The observational AIPW estimator adapter (PRD-004 §10): the deterministic §10.2 cross-fit
# assignment, the fold loop whose every training read is receipted for wall 6, the one hand-written
# AIPW score with influence-function variance (§10.1, §10.4), the §10.5 diagnostic harvest read off
# the same in-memory arrays. Nuisance models live and die
# inside one fold: no sklearn object is ever serialized, and no prediction, weight, or influence
# value leaves this module except inside a restricted object payload.

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Final, NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn import ensemble, linear_model  # type: ignore[import-untyped]

from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1, NuisanceProfileV1
from causal.analysis.methods.aipw import diagnostics
from causal.analysis.methods.randomized import estimation as rct
from causal.shared.contracts import ArtifactRef, decode_contrast

ADAPTER_VERSION: Final = "observational-aipw-adapter.v1"
ASSIGNMENT_ALGORITHM: Final = "seeded_row_hash_stratified.v1"
# The §10.4 bound is a numerical guard carrying its own version, reported apart from any trimming.
BOUND_RULE: Final = "numerical-propensity-bound.v1"
WEIGHT_COLUMN, OVERLAP_BINS = "aipw_implied_weight", 10
FOLD_WITHOUT_BOTH_STATES: Final = "fold_without_both_treatment_states"
NO_CONTRIBUTING, UNREGISTERED_LEARNER = "no_contributing_treated_units", "unregistered_learner"
# Roles the score reads itself; every other declared role column is an adjustment covariate.
RESERVED: Final = ("treatment", "outcome", "unit_identifier", "cluster", "stratum", "time")
_Floats = NDArray[np.float64]
_Bits = NDArray[np.bool_]
_Folds = NDArray[np.int64]
# One fitted preprocessing recipe: training-fitted parameters applied to any matrix.
Transform = Callable[[_Floats], _Floats]


def _number(params: Mapping[str, ec.ParameterValue], key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None or isinstance(value, bool | str) else float(value)


def _glm(hyper: ec.ValueMap) -> Any:
    return linear_model.LogisticRegression(
        C=_number(hyper, "regularization_c", 1.0), max_iter=int(_number(hyper, "max_iter", 1000)),
        random_state=int(_number(hyper, "random_state", 0)))


def _boosted(hyper: ec.ValueMap, binary: bool) -> Any:
    kind = (ensemble.HistGradientBoostingClassifier if binary
            else ensemble.HistGradientBoostingRegressor)
    return kind(max_iter=int(_number(hyper, "max_iter", 200)), early_stopping=False,
                learning_rate=_number(hyper, "learning_rate", 0.1),
                max_leaf_nodes=int(_number(hyper, "max_leaf_nodes", 31)),
                random_state=int(_number(hyper, "random_state", 0)))


# Every registered §10.3 learner id and the pinned sklearn estimator it builds. A binary outcome
# takes the classifier sibling of the registered regressor; nothing else varies at run time.
LEARNERS: Final[dict[str, Callable[[ec.ValueMap, bool], Any]]] = {
    "sklearn_logistic_regression_l2": lambda hyper, binary: _glm(hyper),
    "sklearn_ridge_regression_l2": lambda hyper, binary: _glm(hyper) if binary else
    linear_model.Ridge(alpha=1.0 / _number(hyper, "regularization_c", 1.0)),
    "sklearn_hist_gradient_boosting_classifier": lambda hyper, binary: _boosted(hyper, True),
    "sklearn_hist_gradient_boosting_regressor": _boosted}


def _learner(learner_id: str, hyper: ec.ValueMap, binary: bool) -> Any:
    if (build := LEARNERS.get(learner_id)) is None:
        raise ec.EstimationError(f"no registered learner {learner_id!r}", UNREGISTERED_LEARNER)
    return build(hyper, binary)


def _standardize(train: _Floats) -> Transform:
    scale = np.where(train.std(axis=0) > 0.0, train.std(axis=0), 1.0)
    return lambda matrix: (matrix - train.mean(axis=0)) / scale


# The registered `cross_fit_training_fold` recipes PRD-003 may hand PRD-004 (§10.2 step 1). A plan
# that declares none carries the registered identity, which fits nothing and changes nothing.
RECIPES: Final[dict[str, Callable[[_Floats], Transform]]] = {name: _standardize for name in (
    "estimator_scoped_recipe", "numeric_standardization", "cross_fit_training_fold")}


def covariates(roles: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(column for role, column in roles.items() if role.partition("__")[0]
                        not in RESERVED))


class Data(NamedTuple):
    # One frozen estimator input as arrays: the approved adjustment matrix, the outcome, the
    # treatment indicator, and whether the outcome is binary.
    features: _Floats
    outcome: _Floats
    treated: _Bits
    binary: bool


def frame_data(view: pl.DataFrame, roles: Mapping[str, str], contrast_id: str, *,
               outcome_kind: str | None = None) -> Data:
    # The approved contrast names its exact treated and comparator states. A numeric covariate
    # arrives as itself, any other as its frozen level index, and a null cell as zero beside its
    # approved missingness indicator: PRD-004 invents no imputation of its own (§10.1, §10.5).
    arm = view[roles["treatment"]].cast(pl.String)
    states = sorted(set(arm.drop_nulls().to_list()))
    if len(states) != 2:
        raise ec.EstimationError("AIPW needs exactly two treatment states", NO_CONTRIBUTING)
    treated, _ = rct._pair(contrast_id, states)
    built = [(row.cast(pl.Float64, strict=False) if row.dtype.is_numeric() else row.cast(
        pl.String).cast(pl.Categorical).to_physical().cast(pl.Float64)).fill_null(
        0.0).to_numpy().astype(np.float64) for row in (view[name] for name in covariates(roles))]
    outcome = view[roles["outcome"]].cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()
    return Data(np.column_stack(built) if built else np.zeros((view.height, 1)),
                outcome.astype(np.float64),
                (arm == treated).to_numpy().astype(np.bool_),
                (outcome_kind == "binary" if outcome_kind is not None
                 else set(np.unique(outcome).tolist()) <= {0.0, 1.0}))


def assign_folds(unit_ids: Sequence[str], treated: _Bits, seed: int, count: int) -> _Folds:
    # §10.2: treatment-stratified K folds, dealt inside each treatment state in seeded-hash order,
    # so every fold carries both states whenever the frozen rows do. The same seed over the same
    # row ids always deals the same folds, on any machine and in any process (§13).
    keys = [hashlib.blake2b(f"{seed}:{row}".encode(), digest_size=8).digest() for row in unit_ids]
    folds = np.zeros(len(unit_ids), dtype=np.int64)
    for state in (False, True):
        rows = sorted(np.flatnonzero(treated == state).tolist(), key=lambda row: keys[row])
        folds[rows] = np.arange(len(rows)) % count
    return folds


def fold_counts(folds: _Folds, treated: _Bits, count: int) -> dict[str, ec.CountMap]:
    # Train, validation, and treatment counts by fold: what wall 6 measures a receipt against.
    return {f"fold_{fold}": {"train": int(np.count_nonzero(folds != fold)),
                             "validation": int(np.count_nonzero(folds == fold)),
                             "treated": int(np.count_nonzero(treated[folds != fold])),
                             "control": int(np.count_nonzero(~treated[folds != fold]))}
            for fold in range(count)}


def mapping_object_payload(folds: _Folds, *, seed: int, count: int) -> dict[str, object]:
    # The row-to-fold mapping is a restricted object payload; no envelope or event carries it.
    return {"assignment_algorithm_id": ASSIGNMENT_ALGORITHM, "fold_count": count, "seed": seed,
            "row_count": int(folds.size), "builder_version": ADAPTER_VERSION,
            "fold_by_row_hex": folds.astype(np.uint8).tobytes().hex()}


@dataclass
class Ledger:
    # §10.2, the leakage contract made structural: a fold fit reaches its rows ONLY through `read`,
    # which receipts what it handed over, and `held` returns held-out FEATURES alone, so no
    # validation outcome is reachable from inside a fold fit at all. Wall 6 reads the receipts.

    data: Data
    folds: _Folds
    receipts: dict[str, dict[str, int]] = field(default_factory=dict)

    def read(self, fold: int, scope: str = "train") -> Data:
        rows = np.ones(self.folds.size, np.bool_) if scope == "all" else self.folds != fold
        held = self.folds == fold
        seen = self.receipts.setdefault(f"fold_{fold}", {"train": 0, "validation": 0})
        for name, taken in (("train", rows & ~held), ("validation", rows & held)):
            seen[name] = max(seen[name], int(np.count_nonzero(taken)))
        return Data(self.data.features[rows], self.data.outcome[rows],
                    self.data.treated[rows], self.data.binary)

    def held(self, fold: int) -> _Floats:
        return np.asarray(self.data.features[self.folds == fold], dtype=np.float64)


# What one fold predicts for its held-out rows — propensity, outcome under each treatment state —
# and whether its fits converged.
Held = tuple[_Floats, _Floats, _Floats, bool]
FoldFitter = Callable[[Ledger, int, NuisanceProfileV1, Sequence[str]], Held]


def _predicted(model: Any, features: _Floats, hyper: ec.ValueMap) -> tuple[_Floats, bool]:
    found = (model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba")
             else model.predict(features))
    # A learner that reports no iteration count solved its fit directly and converged by doing so.
    return (np.asarray(found, dtype=np.float64), bool(np.all(np.asarray(
        getattr(model, "n_iter_", None) or [0]) < int(_number(hyper, "max_iter", 1000)))))


def fit_fold(ledger: Ledger, fold: int, profile: NuisanceProfileV1,
             recipe_ids: Sequence[str]) -> Held:
    # §10.2 steps 1-4 for one fold: fit the recipe on training rows only, transform both sides with
    # those fitted parameters, fit the registered propensity and per-state outcome learners on
    # training rows, and predict the held-out rows. Nothing here can read a held-out outcome.
    train, hyper = ledger.read(fold), dict(profile.hyperparameters)
    if not train.treated.any() or bool(train.treated.all()):
        raise ec.EstimationError(f"fold {fold} trains on one state", FOLD_WITHOUT_BOTH_STATES)
    shape = next((RECIPES[name] for name in recipe_ids if name in RECIPES), None)
    fit = shape(train.features) if shape is not None else (lambda matrix: matrix)
    fitted, out = fit(train.features), fit(ledger.held(fold))
    propensity, converged = _predicted(_learner(profile.propensity_learner, hyper, True).fit(
        fitted, train.treated.astype(np.int64)), out, hyper)
    outcomes = [_predicted(_learner(profile.outcome_learner, hyper, train.binary).fit(
        fitted[train.treated == state], train.outcome[train.treated == state]), out, hyper)
        for state in (False, True)]
    return (propensity, outcomes[0][0], outcomes[1][0],
            converged and outcomes[0][1] and outcomes[1][1])


class Score(NamedTuple):
    # The §10.1 score. `influence` is the centered influence function, `weights` the estimator's
    # implied weighting, and `bounded_rows` how many rows the §10.4 guard touched; the first two
    # are restricted arrays that reach no envelope, event, or diagnostic value.
    estimate: float
    standard_error: float
    lower: float
    upper: float
    p_value: float
    influence: _Floats
    weights: _Floats
    bounded_rows: int


def score(data: Data, propensity: _Floats, under_control: _Floats, under_treated: _Floats, *,
          estimand: str, bound: float, level: float) -> Score:
    # The ONE hand-written formula: the AIPW influence function for the approved estimand, its
    # mean, and the influence-function variance. `bound` is the §10.4 division-by-zero guard and
    # never a trimming rule: it is versioned, and the rows it touched are counted and reported.
    treated, guarded = data.treated.astype(np.float64), np.clip(propensity, bound, 1.0 - bound)
    control_gap, treated_gap = data.outcome - under_control, data.outcome - under_treated
    if estimand == "att":
        share, odds = float(treated.mean()), guarded / (1.0 - guarded)
        raw = treated * control_gap - (1.0 - treated) * odds * control_gap
        estimate = float(raw.mean()) / share if share else 0.0
        influence = (raw - treated * estimate) / (share or 1.0)
        weights = treated + (1.0 - treated) * odds
    else:
        psi = (under_treated - under_control + treated * treated_gap / guarded
               - (1.0 - treated) * control_gap / (1.0 - guarded))
        estimate = float(psi.mean())
        influence, weights = psi - estimate, treated / guarded + (1.0 - treated) / (1.0 - guarded)
    error, normal = math.sqrt(float((influence**2).sum())) / max(influence.size, 1), NormalDist()
    edge = normal.inv_cdf(0.5 + level / 2.0) * error
    found = 2.0 * normal.cdf(-abs(estimate) / error) if error > 0.0 else 1.0
    return Score(estimate, error, estimate - edge, estimate + edge, min(max(found, 0.0), 1.0),
                 influence, weights, int(np.count_nonzero(propensity != guarded)))


@dataclass(frozen=True)
class CrossFitRun:
    # Everything one cross-fitted fit produced. The arrays here are restricted: they reach a store
    # as one prediction object and never an envelope, an event, or a diagnostic value (§10.2).

    data: Data
    folds: _Folds
    fold_count: int
    seed: int
    profile_id: str
    counts_by_fold: dict[str, ec.CountMap]
    receipts: dict[str, dict[str, int]]
    predictions: Held
    converged_folds: int
    score: Score

    @property
    def leaked_rows(self) -> int:
        return sum(int(row.get("validation", 0)) for row in self.receipts.values())

    def prediction_payload(self) -> dict[str, object]:
        # The §10.2 private prediction object: typed float64 arrays, never a pickled estimator.
        arrays = zip(("propensity", "under_control", "under_treated", "influence", "weight"),
                     (*self.predictions[:3], self.score.influence, self.score.weights),
                     strict=True)
        return {"builder_version": ADAPTER_VERSION, "dtype": "float64", "row_count": int(
            self.folds.size), "nuisance_profile_id": self.profile_id} | {
            f"{name}_hex": np.asarray(row, np.float64).tobytes().hex() for name, row in arrays}

    def mapping_payload(self) -> dict[str, object]:
        return mapping_object_payload(self.folds, seed=self.seed, count=self.fold_count)

    def assignment(self, plan: ec.EstimationPlanV1, mapping_object: ec.ObjectRefV1, *,
                   plan_ref: ArtifactRef,
                   parents: tuple[ArtifactRef, ...]) -> ec.CrossFitAssignmentV1:
        # The §10.2 assignment artifact: what was dealt, by which algorithm, under which recipe
        # and nuisance profile. It records the deal; it never re-deals one.
        rule = plan.fold_assignment_rule_id
        return ec.CrossFitAssignmentV1(
            parents=parents, versions=dict(plan.versions), plan=plan_ref,
            fold_count=self.fold_count, assignment_algorithm_id=ASSIGNMENT_ALGORITHM,
            mapping_object=mapping_object, counts_by_fold=dict(self.counts_by_fold),
            stratification_rule_ids=(rule,) if rule else (), nuisance_profile_id=self.profile_id,
            preprocessing_recipe_version=",".join(plan.preprocessing_recipe_ids) or "identity")


def cross_fit(view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
              params: ec.ValueMap, fitter: FoldFitter = fit_fold) -> CrossFitRun:
    # One cross-fitted run: deal the folds from the plan seed and the frozen row ids, walk them in
    # order, and assemble the out-of-fold predictions every downstream number is read from. A
    # prespecified branch differs from the primary run only by the parameters it was handed.
    roles, converged = dict(plan.role_columns), 0
    data = frame_data(view, roles, plan.contrast_ids[0],
                      outcome_kind=str(params["outcome_kind"]) if "outcome_kind" in params else None)
    count = int(_number(params, "fold_count", plan.fold_count or pack.fold_count_default or 5))
    seed = plan.seed + int(_number(params, "seed_offset", 0.0))
    unit_ids = [str(value) for value in view[roles["unit_identifier"]].to_list()]
    ledger = Ledger(data, assign_folds(unit_ids, data.treated, seed, count))
    named = params.get("nuisance_profile_id") or plan.nuisance_profile_id
    profile = next((row for row in pack.nuisance_profiles if row.profile_id == named),
                   pack.primary_nuisance_profile())
    if profile is None:
        raise ec.EstimationError(f"{pack.method_id} fits no such profile", UNREGISTERED_LEARNER)
    predicted = [np.zeros(data.treated.size) for _ in range(3)]
    for fold in range(count):
        held = fitter(ledger, fold, profile, plan.preprocessing_recipe_ids)
        for array, values in zip(predicted, held[:3], strict=True):
            array[ledger.folds == fold] = values
        converged += int(held[3])
    return CrossFitRun(
        data, ledger.folds, count, seed, profile.profile_id,
        fold_counts(ledger.folds, data.treated, count), ledger.receipts,
        (predicted[0], predicted[1], predicted[2], converged == count), converged,
        score(data, *predicted, estimand=str(params.get("estimand") or "ate"),
              bound=_number(params, "propensity_bound", 0.01), level=plan.confidence_level))


def primary_item(plan: ec.EstimationPlanV1, params: ec.ValueMap, run: CrossFitRun,
                 unit_ids: pl.Series, mask: ArtifactRef) -> ec.PrimaryContrastResultV1:
    # One primary contrast for the approved estimand, carrying the fold count, the bound rule, and
    # the effective sample the score leaned on among its method quantities (§6.3, §10.1).
    estimand, found, data = str(params.get("estimand") or "ate"), run.score, run.data
    return ec.PrimaryContrastResultV1(
        contrast_id=plan.contrast_ids[0], estimand_id=plan.estimand_id,
        estimand_label=f"{estimand.upper()}, augmented inverse-probability weighting",
        estimate=found.estimate, effect_direction="treated_minus_comparator",
        estimate_units="risk_difference" if data.binary else plan.outcome_scale,
        comparator_id=decode_contrast(plan.contrast_ids[0])[1],
        standard_error=found.standard_error, confidence_level=plan.confidence_level,
        interval_lower=found.lower, interval_upper=found.upper, p_value=found.p_value,
        uncertainty_method=plan.uncertainty_method,
        finite_sample_correction=plan.finite_sample_correction,
        contributing_counts={"row": int(data.treated.size), "unit": unit_ids.n_unique(),
                             "treated": int(np.count_nonzero(data.treated))},
        contribution_mask=mask, estimator_id=plan.estimator_id,
        estimator_version=plan.estimator_version, estimator_parameters=dict(params),
        adapter_version=ADAPTER_VERSION,
        convergence="converged" if run.converged_folds == run.fold_count else "not_converged",
        method_quantities={
            "estimand": estimand, "fold_count": run.fold_count, "seed": run.seed,
            "nuisance_profile_id": run.profile_id, "propensity_bound_rule_id": BOUND_RULE,
            "propensity_bound": _number(params, "propensity_bound", 0.01),
            "propensity_bound_rows": found.bounded_rows})


@dataclass(frozen=True)
class ObservationalAipwAdapter:
    # §19.1: the adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else. `mask` is
    # the committed §6.2 contribution mask every item it reports contributed through, and `fitter`
    # is the registered fold fit, replaceable only to prove the wall-6 receipts bite.

    mask: ArtifactRef
    fitter: FoldFitter = fit_fold

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None, *,
            compute_diagnostics: bool = True) -> engine.AdapterResult:
        # One cross-fitted fit inside one adapter call, then the §10.5 harvest off the same arrays.
        # An override is only ever a prespecified branch delta, never a choice made after seeing a
        # result. The run itself is handed back so a coordinator can commit its §10.2 receipts.
        params, roles = dict(plan.estimator_parameters) | dict(overrides or {}), dict(
            plan.role_columns)
        run = cross_fit(view, plan, pack, params, self.fitter)
        stated = next((row.threshold_params for row in pack.required_diagnostics
                       if row.diagnostic_id == "propensity_common_support"), {})
        return engine.AdapterResult(
            (primary_item(plan, params, run, view[roles["unit_identifier"]], self.mask),),
            diagnostics.harvest(view, roles, plan, run, stated) if compute_diagnostics else {}, run)

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]:
        # The registered Holm step-down is shared machinery, not one pack's property (§9.1).
        return rct.holm(items, level)

