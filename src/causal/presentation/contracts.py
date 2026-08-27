# Presentation catalog, context manifest, curator context, figure plan and specification,
# render, bundle, outcome, and run payloads (PRD-005 §1, §7.1, §8, §10, §13, §18).

from __future__ import annotations

from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef, Identity, PayloadLocator, Sha256Hex, UtcTimestamp

# §1: the closed presentation status set; nothing else terminates the stage.
PresentationOutcomeStatus = Literal[
    "complete", "complete_with_qualifications", "blocked", "needs_template",
    "needs_layout_revision", "failed", "failed_observability"]
DELIVERED_STATUSES: Final = ("complete", "complete_with_qualifications")
# The two claim statuses §5 condition 2 admits; PRD-005 rehabilitates no other.
PresentableClaimStatus = Literal["reportable", "reportable_with_qualifications"]
# §13 pinned display profile and font, and the §8 catalog-wide V1 ceilings.
DISPLAY_PROFILE_ID: Final = "desktop-736-v1"
FONT_ID: Final = "noto-sans-v2.015-variable-normal"
FONT_SHA256: Final = "bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d"
MAX_FIGURES, MAX_PANELS, MAX_CONCURRENCY = 6, 3, 8
# The five §5 handoff entries, keyed by the manifest field each one fills.
ENTRY_KEYS: Final = ("estimation_bundle", "claim_judgment", "figure_data_bundle",
                     "experiment_design", "capacity_check")
# §7.1: the approved semantic selection the manifest copies verbatim from the handoff.
APPROVED_KEYS: Final = ("question_id", "estimand_id", "method_id", "profile_id")
# §8: the enumerated choice families every template declares for the curator.
CHOICE_KEYS: Final = ("marks", "axes", "scale_sharing", "labels", "legends", "qualifications",
                      "reference_lines")
# §14: the accessible text every figure carries, drafted once and compiled unchanged.
TEXT_KEYS: Final = ("title", "caption", "accessible_description")

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_NonNegInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(ge=1)]
_Ids = Annotated[tuple[Identity, ...], Field(min_length=1)]
_Text = Annotated[str, Field(min_length=1, max_length=600)]
_Prose = Annotated[str, Field(max_length=4000)]


class PresentationError(ValueError):
    # `code` is stable; `detail_codes` carries every failing family (EstimationError idiom).

    def __init__(self, message: str, code: str, detail_codes: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code, self.detail_codes = code, detail_codes


# Frozen, strict base for presentation row and fragment models.
class _Row(BaseModel):
    model_config = _MODEL_CONFIG


# Base for committed presentation payloads.
class _Payload(_Row):
    def canonical_payload(self) -> dict[str, Any]:
        # Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable.
        return self.model_dump(mode="json")


# Ordered upstream lineage and the pinned registry versions, on every committed payload.
class _Lineage(_Payload):
    parents: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    versions: dict[str, Identity]


# §13: the one pinned display profile. Every number is fixed here, never configured at run time.
class DisplayProfileV1(_Row):
    display_profile_id: Literal["desktop-736-v1"] = "desktop-736-v1"
    display_profile_version: Identity
    logical_width: Literal[736] = 736
    outer_padding: Literal[24] = 24
    png_width: Literal[1472] = 1472
    width_axis: Literal[100] = 100
    font_weights: Annotated[tuple[Literal[400, 600, 700], ...], Field(min_length=1)]


# §8: every capacity a template declares is finite. An absent, non-positive, or unordered bound
# fails validation, so no registered template can ever claim unbounded capacity.
class TemplateBoundsV1(_Row):
    min_panels: _PositiveInt
    max_panels: _PositiveInt
    min_series_per_panel: _PositiveInt
    max_series_per_panel: _PositiveInt
    max_series_per_figure: _PositiveInt
    max_labels: _PositiveInt
    max_label_characters: _PositiveInt
    max_annotations_per_panel: _NonNegInt
    max_annotations_per_figure: _NonNegInt
    min_logical_height: _PositiveInt
    max_logical_height: _PositiveInt

    @model_validator(mode="after")
    def _every_bound_is_ordered(self) -> Self:
        pairs = ((self.min_panels, self.max_panels),
                 (self.min_series_per_panel, self.max_series_per_panel),
                 (self.max_annotations_per_panel, self.max_annotations_per_figure),
                 (self.min_logical_height, self.max_logical_height))
        if any(low > high for low, high in pairs):
            raise ValueError("a template bound states its minimum before its maximum")
        return self


# One registered template: what it may draw, for which profiles, and exactly how much (§8).
class TemplateV1(_Row):
    template_id: Identity
    template_version: Identity
    allowed_profile_ids: _Ids
    visual_evidence_ids: _Ids
    figure_data_schema_ids: _Ids
    field_mappings: dict[str, Identity]
    # The enumerated legal choices, keyed by CHOICE_KEYS: marks, axes, scale sharing, label and
    # legend positions, qualification placements, and reference lines (§8, §9.1).
    choices: dict[str, _Ids]
    bounds: TemplateBoundsV1
    # Synthetic fixture id to its content hash, and to the semantic result it must produce.
    fixtures: dict[str, Sha256Hex]
    fixture_results: dict[str, Literal["valid", "needs_layout_revision", "needs_template"]]


# One method profile: the §12 evidence questions and the templates that answer them honestly.
class MethodProfileV1(_Row):
    profile_id: Identity
    method_id: Identity
    required_evidence_ids: _Ids
    conditional_evidence_ids: tuple[Identity, ...]
    evidence_order: _Ids
    templates_by_evidence: dict[str, _Ids]
    permitted_panel_combinations: tuple[_Ids, ...]
    # Mandatory reference lines and uncertainty fields, keyed by evidence id (§8).
    mandatory_encodings: dict[str, _Ids]
    qualification_placement: Identity
    # Exact method-level ceilings for evidence families, primary items, diagnostic and
    # sensitivity groups (§8).
    capacity_limits: dict[str, _PositiveInt]

    def questions(self) -> set[str]:
        return set(self.required_evidence_ids) | set(self.conditional_evidence_ids)


# §8 catalog-wide V1 bounds. A catalog may tighten these and may never widen them.
class CatalogLimitsV1(_Row):
    max_figures: Annotated[int, Field(ge=1, le=MAX_FIGURES)]
    max_panels_per_figure: Annotated[int, Field(ge=1, le=MAX_PANELS)]
    max_concurrent_tasks: Annotated[int, Field(ge=1, le=MAX_CONCURRENCY)]
    accessible_table_max_rows: _PositiveInt


# §8: one immutable catalog per run — profiles, templates, the display profile, the theme, the
# vendored font identity, and the capacity ceilings. The curator cannot add to or edit it.
class VisualizationCatalogV1(_Payload):
    schema_version: Literal["visualization-catalog.v1"] = "visualization-catalog.v1"
    catalog_id: Identity
    catalog_version: Identity
    implementation_version: Identity
    created_at: UtcTimestamp
    theme_id: Identity
    theme_version: Identity
    theme_hash: Sha256Hex
    font_id: Literal["noto-sans-v2.015-variable-normal"]
    font_sha256: Sha256Hex
    font_source_commit: Identity
    font_file_path: PayloadLocator
    display_profile: DisplayProfileV1
    limits: CatalogLimitsV1
    profiles: Annotated[tuple[MethodProfileV1, ...], Field(min_length=1)]
    templates: Annotated[tuple[TemplateV1, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _every_reference_resolves_inside_the_catalog(self) -> Self:
        known = {row.template_id: row for row in self.templates}
        profiles = {row.profile_id for row in self.profiles}
        if len(known) != len(self.templates) or len(profiles) != len(self.profiles):
            raise ValueError("a catalog registers each template and profile id exactly once")
        for row in self.templates:
            if (set(row.allowed_profile_ids) - profiles or set(row.choices) != set(CHOICE_KEYS)
                    or set(row.fixtures) != set(row.fixture_results)
                    or row.bounds.max_panels > self.limits.max_panels_per_figure):
                raise ValueError(f"{row.template_id} is not a legal entry of this catalog")
        for profile in self.profiles:
            if profile.questions() - set(profile.templates_by_evidence) or (
                    profile.questions() - set(profile.evidence_order)):
                raise ValueError(f"{profile.profile_id} leaves an evidence question unanswered")
            for evidence_id, ids in profile.templates_by_evidence.items():
                if not all(name in known and profile.profile_id in known[name].allowed_profile_ids
                           and evidence_id in known[name].visual_evidence_ids for name in ids):
                    raise ValueError(f"{profile.profile_id} cannot answer {evidence_id} that way")
        return self


# One evidence question as the manifest froze it: bounded facts, never a raw observation (§7.1).
class EvidenceEntryV1(_Row):
    visual_evidence_id: Identity
    figure_data: ArtifactRef
    figure_data_schema_id: Identity
    compatible_template_ids: _Ids
    # Quantity and unit ids, and the short description of each referenced frozen field (§7.1).
    quantities: dict[str, Identity]
    units: dict[str, Identity]
    cardinality: _NonNegInt
    suppression_state: Identity


# The approved claim surface and its bounded evidence, shared by the manifest and the payload
# the curator receives; neither may carry a statement the claim judgment did not freeze (§7.1).
class _ApprovedClaims(_Row):
    claim_status: PresentableClaimStatus
    statement_ids: _Ids
    qualification_ids: tuple[Identity, ...]
    evidence: Annotated[tuple[EvidenceEntryV1, ...], Field(min_length=1)]


# §7.1: the one authoritative presentation-context surface, frozen before the curator runs.
class PresentationContextManifestV1(_ApprovedClaims, _Lineage):
    schema_version: Literal["presentation-context-manifest.v1"] = "presentation-context-manifest.v1"
    analysis_id: Identity
    stage_run_id: Identity
    handoff_manifest: ArtifactRef
    inputs: dict[str, ArtifactRef]
    causal_graph_view: ArtifactRef
    # The approved question, estimand, method, and catalog profile, keyed by APPROVED_KEYS.
    approved: dict[str, Identity]
    required_evidence_ids: _Ids
    display_profile: DisplayProfileV1
    allowlists: dict[str, _Ids]

    @model_validator(mode="after")
    def _the_five_entries_and_every_required_question_are_present(self) -> Self:
        if set(self.inputs) != set(ENTRY_KEYS) or set(self.approved) != set(APPROVED_KEYS):
            raise ValueError("a manifest names the five entries and one approved selection")
        if set(self.required_evidence_ids) - {row.visual_evidence_id for row in self.evidence}:
            raise ValueError("a manifest carries every required visual-evidence question")
        return self


# §7.1 curator payload: approved claims, bounded evidence facts, and legal choices only.
class PresentationCuratorContextV1(_ApprovedClaims):
    manifest: ArtifactRef
    templates: Annotated[tuple[TemplateV1, ...], Field(min_length=1)]
    display_profile: DisplayProfileV1
    limits: CatalogLimitsV1


# One figure: a template choice, an ordering, and enumerated placement choices (§9.1, §10).
class FigureEntryV1(_Row):
    figure_id: Identity
    template_id: Identity
    visual_evidence_ids: _Ids
    panel_groups: Annotated[tuple[_Ids, ...], Field(min_length=1, max_length=MAX_PANELS)]
    # The enumerated scale-sharing, label-position, and legend-position choices taken.
    choices: dict[str, Identity]
    annotation_ids: tuple[Identity, ...]
    qualification_ids: tuple[Identity, ...]
    # Title, caption, and accessible description, keyed by TEXT_KEYS (§14).
    text: dict[str, _Text]


# §9.3: one draft per curator call — either an ordered plan or one typed inability.
class FigurePlanDraftV1(_Row):
    figures: Annotated[tuple[FigureEntryV1, ...], Field(max_length=MAX_FIGURES)] = ()
    summary: _Prose = ""
    inability_code: Identity | None = None
    implicated_evidence_ids: tuple[Identity, ...] = ()

    @model_validator(mode="after")
    def _a_draft_is_a_plan_or_a_typed_inability(self) -> Self:
        if bool(self.figures) == (self.inability_code is not None):
            raise ValueError("a draft carries figures or one inability code, never both")
        if any(set(row.text) != set(TEXT_KEYS) for row in self.figures):
            raise ValueError("every drafted figure carries its title, caption, and description")
        return self


# §10: the accepted plan, frozen before compilation. It holds no number of its own.
class FigurePlanV1(FigurePlanDraftV1, _Lineage):
    schema_version: Literal["figure-plan.v1"] = "figure-plan.v1"
    plan_id: Identity
    # Required evidence id to the one figure covering it, and evidence id to its frozen data.
    coverage: dict[str, Identity]
    figure_data: dict[str, ArtifactRef]

    @model_validator(mode="after")
    def _coverage_names_one_distinct_committed_figure(self) -> Self:
        ids = [row.figure_id for row in self.figures]
        if not ids or len(set(ids)) != len(ids) or set(self.coverage.values()) - set(ids):
            raise ValueError("every covered evidence question names one distinct figure")
        return self


# One axis: a referenced quantity with a mechanically computed domain (§11.2, §13).
class AxisSpecV1(_Row):
    quantity_id: Identity
    unit_id: Identity
    title: _Text
    scale: Literal["linear", "log", "band", "time"]
    domain: tuple[float, float] | None


# One panel answering exactly one question (§11.1).
class PanelSpecV1(_Row):
    panel_id: Identity
    question: _Text
    mark: Identity
    axes: dict[str, AxisSpecV1]
    # Series, referenced frozen fields, uncertainty encoding, reference lines, and annotations,
    # each keyed by its role (§13).
    encodings: dict[str, _Ids]


# §13: one declarative specification per accepted figure; its canonical hash is its identity.
class FigureSpecV1(_Lineage):
    schema_version: Literal["figure-spec.v1"] = "figure-spec.v1"
    figure_id: Identity
    template_id: Identity
    panels: Annotated[tuple[PanelSpecV1, ...], Field(min_length=1, max_length=MAX_PANELS)]
    text: dict[str, _Text]
    qualification_ids: tuple[Identity, ...]
    logical_height: _PositiveInt

    def spec_hash(self) -> str:
        # §13.1: the exact identity of this specification, equal in every environment.
        return content_hash(self.canonical_payload())


# §13: one SVG and one PNG from the same specification, plus the exact renderer fingerprint.
class RenderArtifactV1(_Lineage):
    schema_version: Literal["presentation-render.v1"] = "presentation-render.v1"
    figure_id: Identity
    spec_hash: Sha256Hex
    # The `svg` and `png` locators, and the exact hash of each rendered byte stream.
    objects: dict[str, PayloadLocator]
    object_hashes: dict[str, Sha256Hex]
    logical_height: _PositiveInt
    renderer_fingerprint: dict[str, Identity]


# §1: the one immutable presentation product; every figure carries its render and its table.
class PresentationBundleV1(_Lineage):
    schema_version: Literal["presentation-bundle.v1"] = "presentation-bundle.v1"
    context_manifest: ArtifactRef
    plan: ArtifactRef
    specs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1, max_length=MAX_FIGURES)]
    renders: Annotated[tuple[ArtifactRef, ...], Field(min_length=1, max_length=MAX_FIGURES)]
    accessible_tables: tuple[ArtifactRef, ...]
    causal_graph_view: ArtifactRef
    validation_report: ArtifactRef
    summary: _Prose


# §1: the presentation stage's single terminal record, over the closed status set.
class PresentationOutcomeV1(_Payload):
    schema_version: Literal["presentation-outcome.v1"] = "presentation-outcome.v1"
    status: PresentationOutcomeStatus
    stage_run_id: Identity
    context_manifest: ArtifactRef | None
    presentation_bundle: ArtifactRef | None
    error_code: Identity | None
    detail_codes: tuple[Identity, ...] = ()

    @model_validator(mode="after")
    def _only_a_delivered_outcome_carries_a_bundle(self) -> Self:
        if (self.status in DELIVERED_STATUSES) != (self.presentation_bundle is not None):
            raise ValueError(f"a {self.status} outcome carries no presentation bundle")
        return self


# §18: the small run record. The immutable artifacts, not this row, are the source of truth.
class PresentationRunV1(_Payload):
    schema_version: Literal["presentation-run.v1"] = "presentation-run.v1"
    analysis_id: Identity
    stage_run_id: Identity
    status: PresentationOutcomeStatus
    # The five upstream artifacts, and the incoming and outgoing handoff manifests plus the
    # manifest, plan, validation, spec, render, and bundle artifacts this run holds.
    upstream: dict[str, ArtifactRef]
    current: dict[str, ArtifactRef]
    # Catalog, display-profile, theme, font, compiler, and renderer ids, hashes, versions.
    identities: dict[str, Identity]
    # Task, attempt, event, correction, and trace-acknowledgement counts, and the stable
    # trace-acknowledgement and failure ids behind them (§18).
    counters: dict[str, _NonNegInt]
    stable_ids: dict[str, tuple[Identity, ...]]
    created_at: UtcTimestamp
    completed_at: UtcTimestamp | None
