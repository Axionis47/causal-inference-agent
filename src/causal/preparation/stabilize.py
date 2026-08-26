"""Row identity and the Phase A ordered disposition engine (PRD-003 §8, §9, §24.4; T-016)."""

from __future__ import annotations

import datetime as dt
import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import polars as pl
from pydantic import Field

from causal.preparation import contracts as pc
from causal.preparation.contracts import PreparationError, RowDisposition, _Row
from causal.preparation.plans import PreparationPackV1
from causal.shared.canonical import canonical_bytes, content_hash
from causal.shared.contracts import Identity, Sha256Hex
from causal.shared.frames import FRAME_DTYPES, FrameObjectStore

UNPARSABLE_SOURCE, UNREGISTERED_RULE = "unparsable_source", "unregistered_rule"
UNKNOWN_RULE_TARGET, UNREPRESENTABLE_CELL = "unknown_rule_target", "unrepresentable_optional_cell"
REQUIRED_IDENTITY_MISSING, REQUIRED_ROLE_MISSING = "required_identity_missing", "required_role_missing"
REQUIRED_ROLE_UNREGISTERED: Final = "required_role_exclusion_unregistered"
EXACT_DUPLICATE_RECORD, KEY_COLLISION_CONFLICT = "exact_duplicate_record", "key_collision_conflict"


class StabilizationError(PreparationError):
    """Stabilization cannot proceed. `code` is a stable contract value."""


class RuleEvaluator(StrEnum):
    """The closed evaluator set a registered row rule may use (§9.2)."""

    SET_MEMBERSHIP = "set_membership"
    NUMERIC_RANGE = "numeric_range"
    DATE_RANGE = "date_range"
    NONNULL = "nonnull"
    TIMEFRAME_WINDOW = "timeframe_window"


_DATE_EVALUATORS: Final = (RuleEvaluator.DATE_RANGE, RuleEvaluator.TIMEFRAME_WINDOW)
_TIMEFRAME: Final = RowDisposition.NOT_ELIGIBLE_TIMEFRAME


# One column of the pinned parser profile; a critical cell decides record usability.
class ColumnParseSpecV1(_Row):
    column_name: Identity
    dtype: Identity
    parse_critical: bool


# One approved eligibility or unusable-row rule, copied verbatim from the contract (§9.2).
class RowRuleV1(_Row):
    rule_id: Identity
    rule_kind: Identity
    evaluator: RuleEvaluator
    column: Identity
    disposition: RowDisposition
    allowed_values: tuple[str, ...] = ()
    minimum: float | str | None = None
    maximum: float | str | None = None


# One method-required observed role and the exclusion rule that backs it (§9.3).
class RequiredRoleRuleV1(_Row):
    role: Identity
    column: Identity
    observed_rule_id: Identity
    disposition_rule_id: Identity | None = None


# The only two duplicate authorizations V1 has; absent means rows stay distinct (§9.4).
class DuplicatePolicyV1(_Row):
    exact_duplicate_rule_id: Identity | None = None
    conflict_resolution_rule_id: Identity | None = None


# One source row's identity, its single primary disposition, and every later condition.
class RowOutcomeV1(_Row):
    row_number: int = Field(ge=1)
    source_row_id: Identity
    row_content_hash: Sha256Hex
    disposition: RowDisposition
    rule_id: Identity | None
    warning_codes: tuple[Identity, ...]


# The typed frame plus the rows the pinned parser could not represent (§24.4).
@dataclass(frozen=True)
class ParsedSource:
    frame: pl.DataFrame
    corrupt_rows: frozenset[int]
    parse_warning_counts: dict[str, int]


# Every row's outcome plus the per-rule counts the eligibility summary needs.
@dataclass(frozen=True)
class StabilizationResult:
    rows: tuple[RowOutcomeV1, ...]
    rule_counts: dict[str, int]

    def retained_mask(self) -> list[bool]:
        return [row.disposition in pc.RETAINING_DISPOSITIONS for row in self.rows]

    def counts(self) -> tuple[pc.DispositionCountV1, ...]:
        tally = Counter(row.disposition for row in self.rows)
        return tuple(pc.DispositionCountV1(disposition=key, row_count=value)
                     for key, value in sorted(tally.items()))


def row_content_hash(values: Sequence[object]) -> str:
    """SHA-256 over the canonical JSON array of one parsed row's values, in column order.

    Dates and times serialize as ISO strings; any other non-JSON cell type raises the shared
    `CanonicalizationError`, so no unrepresentable value can reach a row identity.
    """
    scalars = [v.isoformat() if isinstance(v, (dt.datetime, dt.date, dt.time)) else v
               for v in values]
    return content_hash({"values": scalars})


# The §24.4 row identity: selected CSV hash, 1-based parsed row number, row content hash.
def source_row_id(csv_hash: str, row_number: int, content_digest: str) -> str:
    return f"{csv_hash}:{row_number}:{content_digest}"


# Parse under the pinned profile; typed cast errors, not a byte scanner, mark records.
def parse_source_csv(data: bytes, columns: tuple[ColumnParseSpecV1, ...]) -> ParsedSource:
    try:
        raw = pl.read_csv(io.BytesIO(data), has_header=True, infer_schema_length=0,
                          truncate_ragged_lines=False)
    except pl.exceptions.PolarsError as error:
        raise StabilizationError(f"the pinned parser cannot represent the source: {error}",
                                 UNPARSABLE_SOURCE) from error
    if missing := sorted({column.column_name for column in columns} - set(raw.columns)):
        raise StabilizationError(f"profile columns absent: {missing}", UNPARSABLE_SOURCE)
    corrupt: set[int] = set()
    optional: set[int] = set()
    typed = raw
    for spec in columns:
        if spec.dtype not in FRAME_DTYPES:
            raise StabilizationError(f"dtype outside the profile: {spec.dtype}", UNPARSABLE_SOURCE)
        source, target = raw.get_column(spec.column_name), FRAME_DTYPES[spec.dtype]
        cast = (source.str.strptime(target, strict=False)  # type: ignore[arg-type]
                if target.is_temporal() else source.cast(target, strict=False))
        failed = (source.is_not_null() & cast.is_null()).to_list()
        rows = [number + 1 for number, bad in enumerate(failed) if bad]
        (corrupt if spec.parse_critical else optional).update(rows)
        typed = typed.with_columns(cast.alias(spec.column_name))
    return ParsedSource(frame=typed, corrupt_rows=frozenset(corrupt),
                        parse_warning_counts={UNREPRESENTABLE_CELL: len(optional)} if optional
                        else {})


@dataclass(frozen=True)
class _Condition:
    # One evaluated rule: which rows it hit, what it would assign, and under which id.
    mask: tuple[bool, ...]
    disposition: RowDisposition
    rule_id: str | None
    code: str


# Rows failing one registered rule; a null never satisfies a bound or a set.
def _fail_mask(frame: pl.DataFrame, rule: RowRuleV1) -> pl.Series:
    if rule.column not in frame.columns:
        raise StabilizationError(f"rule {rule.rule_id} names {rule.column}", UNKNOWN_RULE_TARGET)
    column = frame.get_column(rule.column)
    if rule.evaluator is RuleEvaluator.NONNULL:
        return column.is_null()
    if rule.evaluator is RuleEvaluator.SET_MEMBERSHIP:
        member = column.cast(pl.String, strict=False).is_in(list(rule.allowed_values))
        return ~member.fill_null(value=False)
    as_date = rule.evaluator in _DATE_EVALUATORS
    values = column.cast(pl.Date if as_date else pl.Float64, strict=False)
    inside = pl.Series(values=[True] * column.len(), dtype=pl.Boolean)
    for bound, above in ((rule.minimum, True), (rule.maximum, False)):
        if bound is None:
            continue
        limit = dt.date.fromisoformat(str(bound)) if as_date else float(bound)
        side = values >= limit if above else values <= limit
        inside = inside & side.fill_null(value=False)
    return ~inside


# Evaluate population rules before timeframe rules; an unregistered rule fails closed.
def _rule_conditions(frame: pl.DataFrame, rules: tuple[RowRuleV1, ...], pack: PreparationPackV1,
                     vocabulary: tuple[str, ...]) -> list[_Condition]:
    conditions: list[_Condition] = []
    for rule in sorted(rules, key=lambda item: item.disposition is _TIMEFRAME):
        if rule.rule_kind not in vocabulary:
            raise StabilizationError(f"{rule.rule_id} uses unregistered kind {rule.rule_kind}",
                                     UNREGISTERED_RULE)
        if rule.rule_id not in pack.permitted_disposition_rule_ids:
            raise StabilizationError(f"{rule.rule_id} is not permitted by {pack.method_id}",
                                     UNREGISTERED_RULE)
        conditions.append(_Condition(tuple(_fail_mask(frame, rule).to_list()), rule.disposition,
                                     rule.rule_id, rule.rule_id))
    return conditions


# Rows missing any required identity or grain field (§9.1 step 4).
def _identity_condition(frame: pl.DataFrame, key_columns: tuple[str, ...]) -> _Condition:
    if absent := sorted(set(key_columns) - set(frame.columns)):
        raise StabilizationError(f"key columns absent: {absent}", UNKNOWN_RULE_TARGET)
    missing = frame.select(pl.any_horizontal(pl.col(key_columns).is_null())).to_series()
    return _Condition(tuple(missing.to_list()), RowDisposition.UNUSABLE_REQUIRED_IDENTITY, None,
                      REQUIRED_IDENTITY_MISSING)


# The §9.3 four-condition test decides what a missing required-role value costs a row.
def _role_condition(frame: pl.DataFrame, rule: RequiredRoleRuleV1,
                    manifest: pc.PreparationContextManifestV1,
                    pack: PreparationPackV1) -> _Condition:
    if manifest.column_roles.get(rule.column) != rule.role or rule.column not in frame.columns:
        raise StabilizationError(f"no {rule.role} column {rule.column}", UNKNOWN_RULE_TARGET)
    mask = tuple(frame.get_column(rule.column).is_null().to_list())
    required = rule.observed_rule_id in pack.required_observed_role_rules
    non_imputable = rule.role in pack.protected_roles
    registered = rule.disposition_rule_id in pack.permitted_disposition_rule_ids and (
        rule.disposition_rule_id in manifest.unusable_row_rule_ids)
    if required and non_imputable and registered:
        return _Condition(mask, RowDisposition.UNUSABLE_REQUIRED_ROLE, rule.disposition_rule_id,
                          REQUIRED_ROLE_MISSING)
    if required and non_imputable:
        return _Condition(mask, RowDisposition.UNRESOLVED_CONFLICT, rule.disposition_rule_id,
                          REQUIRED_ROLE_UNREGISTERED)
    return _Condition(mask, RowDisposition.RETAINED_WITH_MISSINGNESS, None, REQUIRED_ROLE_MISSING)


# Exact duplicates stay distinct; a conflicting key without a resolution rule conflicts.
def _duplicate_conditions(frame: pl.DataFrame, digests: Sequence[str],
                          key_columns: tuple[str, ...],
                          policy: DuplicatePolicyV1) -> list[_Condition]:
    repeats = (~pl.Series(values=digests, dtype=pl.String).is_first_distinct()).to_list()
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, key in enumerate(frame.select(key_columns).rows()):
        if all(value is not None for value in key):
            groups.setdefault(key, []).append(index)
    conflicting = [False] * frame.height
    resolved = policy.conflict_resolution_rule_id is not None
    for members in groups.values():
        if len({digests[index] for index in members}) > 1:
            for index in members[1:] if resolved else members:
                conflicting[index] = True
    grain = RowDisposition.UNUSABLE_GRAIN_VIOLATION
    exact = grain if policy.exact_duplicate_rule_id else RowDisposition.RETAINED
    collision = grain if resolved else RowDisposition.UNRESOLVED_CONFLICT
    return [
        _Condition(tuple(repeats), exact, policy.exact_duplicate_rule_id, EXACT_DUPLICATE_RECORD),
        _Condition(tuple(conflicting), collision, policy.conflict_resolution_rule_id,
                   KEY_COLLISION_CONFLICT)]


# Run the §9.1 ordered engine: the first terminal rule is primary, later hits are warnings.
def stabilize(parsed: ParsedSource, *, csv_hash: str, manifest: pc.PreparationContextManifestV1,
              pack: PreparationPackV1, rules: tuple[RowRuleV1, ...],
              role_rules: tuple[RequiredRoleRuleV1, ...], duplicates: DuplicatePolicyV1,
              vocabulary: tuple[str, ...]) -> StabilizationResult:
    frame = parsed.frame
    digests = [row_content_hash(values) for values in frame.rows()]
    keys = tuple(manifest.key_columns)
    corrupt = tuple(number + 1 in parsed.corrupt_rows for number in range(frame.height))
    conditions = [
        _Condition(corrupt, RowDisposition.UNUSABLE_CORRUPT_RECORD, None,
                   RowDisposition.UNUSABLE_CORRUPT_RECORD.value),
        *_rule_conditions(frame, rules, pack, vocabulary),
        _identity_condition(frame, keys),
        *(_role_condition(frame, rule, manifest, pack) for rule in role_rules),
        *_duplicate_conditions(frame, digests, keys, duplicates)]
    partial = RowDisposition.RETAINED_WITH_MISSINGNESS
    outcomes: list[RowOutcomeV1] = []
    rule_counts: dict[str, int] = {}
    for index, digest in enumerate(digests):
        hits = [item for item in conditions if item.mask[index]]
        terminal = next(
            (hit for hit in hits if hit.disposition not in pc.RETAINING_DISPOSITIONS), None)
        kept = partial if any(h.disposition is partial for h in hits) else RowDisposition.RETAINED
        primary = terminal or _Condition((), kept, None, "")
        for hit in hits:
            if hit.rule_id is not None:
                rule_counts[hit.rule_id] = rule_counts.get(hit.rule_id, 0) + 1
        outcomes.append(RowOutcomeV1(
            row_number=index + 1, source_row_id=source_row_id(csv_hash, index + 1, digest),
            row_content_hash=digest, disposition=primary.disposition, rule_id=primary.rule_id,
            warning_codes=tuple(hit.code for hit in hits if hit is not terminal)))
    return StabilizationResult(rows=tuple(outcomes), rule_counts=rule_counts)


# Put one canonical JSON payload at `objects/{sha256}` and return its reference.
def _store(objects: FrameObjectStore, payload: dict[str, object]) -> pc.ObjectRefV1:
    digest = content_hash(payload)
    return pc.ObjectRefV1(object_locator=objects.put_if_absent(digest, canonical_bytes(payload)),
                          content_hash=digest)


# The four §24.2 summaries: row index, eligibility, disposition ledger, and row-set freeze.
def stabilization_summaries(
    objects: FrameObjectStore, parsed: ParsedSource, result: StabilizationResult,
    rules: tuple[RowRuleV1, ...], unique_units: int,
) -> tuple[pc.SourceRowIndexSummaryV1, pc.EligibilityEvaluationSummaryV1,
           pc.DispositionLedgerSummaryV1, pc.RowSetFreezeV1]:
    corrupt = RowDisposition.UNUSABLE_CORRUPT_RECORD
    index = _store(objects, {"rows": [
        {"row_number": row.row_number, "source_row_id": row.source_row_id,
         "row_content_hash": row.row_content_hash,
         "parse_status": "corrupt" if row.disposition is corrupt else "parsed"}
        for row in result.rows]})
    ledger = _store(objects, {"rows": [
        {"source_row_id": row.source_row_id, "disposition": row.disposition.value,
         "rule_id": row.rule_id, "warning_codes": list(row.warning_codes)}
        for row in result.rows]})
    keep = result.retained_mask()
    retained = [row.source_row_id for row, kept in zip(result.rows, keep, strict=True) if kept]
    frozen = _store(objects, {"source_row_ids": retained})
    return (
        pc.SourceRowIndexSummaryV1(row_count=len(result.rows), index_object=index,
                                   parse_warning_counts=dict(parsed.parse_warning_counts)),
        pc.EligibilityEvaluationSummaryV1(
            evaluated_row_count=len(result.rows),
            rule_counts={rule.rule_id: result.rule_counts.get(rule.rule_id, 0) for rule in rules}),
        pc.DispositionLedgerSummaryV1(counts=result.counts(), ledger_object=ledger),
        pc.RowSetFreezeV1(retained_row_object=frozen, retained_row_count=len(retained),
                          unique_unit_count=unique_units, row_set_hash=frozen.content_hash),
    )
