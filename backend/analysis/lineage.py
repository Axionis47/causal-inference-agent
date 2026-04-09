"""Deterministic outcome-lineage classification over accepted evidence identities."""

from backend.analysis import authority
from backend.analysis.contracts import Claim, ClaimAdmission, ClaimScope, Roles, SourceExcerpt, Target
from backend.context.contracts import ContextContent, statement_entries


def outcome_paths(context: ContextContent, target: Target, *, scientific_version: int = 8) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Source-established and qualified possible paths, independent of proposed roles."""
    paths: dict[str, set[str]] = {cid: set() for cid in target.outcome.column_ids} if target.outcome else {}
    edges: list[tuple[str, str, str]] = []
    qualified: list[tuple[set[str], set[str], str]] = []
    for rid, relationship in context.relationships.items():
        if (
            relationship.kind not in {"derivation", "measurement"}
            or relationship.basis != "source_statement"
            or relationship.status in {"withdrawn", "unresolved"}
        ):
            continue
        sources = {p.id for p in relationship.participants if p.role == "from"}
        destinations = {p.id for p in relationship.participants if p.role == "to"}
        reference = f"relationships.{rid}"
        if (
            len(sources) != 1 or len(destinations) != 1 or relationship.conditions or relationship.timing or relationship.qualifications
            or scientific_version >= 19 and relationship.status != "reviewed"
            or any(p.role in {"condition", "intermediate", "participant"} for p in relationship.participants)
        ):
            qualified.append((sources or {p.id for p in relationship.participants}, destinations or {p.id for p in relationship.participants}, reference))
        else:
            edges.append((next(iter(sources)), next(iter(destinations)), reference))
    changed = True
    while changed:
        changed = False
        for source, destination, reference in sorted(edges):
            if source in paths:
                refs = paths[source] | {reference}
                if destination not in paths or not refs <= paths[destination]:
                    paths.setdefault(destination, set()).update(refs)
                    changed = True
    uncertain: dict[str, set[str]] = {}
    possible = [(sources, destinations, ref, True) for sources, destinations, ref in qualified]
    possible.extend(({a}, {b}, ref, False) for a, b, ref in edges)
    changed = True
    while changed:
        changed = False
        for sources, destinations, reference, introduces_uncertainty in possible:
            reached = sources & (paths.keys() | uncertain.keys())
            # Only a qualified relationship introduces uncertainty. Ordinary edges propagate it.
            if not reached or (not introduces_uncertainty and not sources & uncertain.keys()):
                continue
            refs = {reference} | set().union(*(paths.get(cid, set()) | uncertain.get(cid, set()) for cid in reached))
            for destination in destinations:
                if destination not in uncertain or not refs <= uncertain[destination]:
                    uncertain.setdefault(destination, set()).update(refs)
                    changed = True
    return paths, uncertain


def classify(value: Roles, context: ContextContent, target: Target, *, scientific_version: int = 8) -> Roles:
    value = value.model_copy(deep=True)
    paths, uncertain = outcome_paths(context, target, scientific_version=scientific_version)
    catalog = dict(statement_entries(context)) if scientific_version >= 19 else {}
    for item in value.assignments:
        refs = set().union(*(paths[cid] for cid in item.variable.column_ids if cid in paths))
        if refs:
            item.outcome_proxy = True
            item.proxy_evidence = Claim(
                statement="Supplied directed measurement/derivation lineage reaches this variable from the target.",
                basis="supplied", references=sorted(refs),
            )
        elif any(cid in uncertain for cid in item.variable.column_ids):
            item.outcome_proxy = None
            item.proxy_evidence = Claim(
                statement="Qualified or multi-party target lineage needs scoped interpretation. It is not flattened into an unconditional derivation; this variable remains withheld.",
                basis="unknown", references=sorted(set().union(*(uncertain.get(cid, set()) for cid in item.variable.column_ids))),
            )
        elif (
            item.timing == "before_treatment" and (authority.established(item.timing_evidence)
                if scientific_version >= 19 else item.timing_evidence.basis == "supplied")
            and item.timing_evidence.references
        ):
            item.outcome_proxy = False
            item.proxy_evidence = Claim(
                statement="Under the accepted supplied pre-treatment measurement account, this is not a target-period measurement. No supplied target derivation contradicts that account.",
                basis="supplied", references=item.timing_evidence.references,
            )
        else:
            item.outcome_proxy = None
            item.proxy_evidence = Claim(
                statement="Target-outcome lineage is not established. Sharing a follow-up period does not establish derivation or leakage. This column is withheld from diagnostics.",
                basis="unknown",
            )
        if scientific_version >= 19:
            # This proof concerns the finite access rule over frozen records, not
            # a claim that missing relationships establish scientific absence.
            rule = "outcome_lineage" if item.outcome_proxy else "baseline_lineage" if item.outcome_proxy is False else "withheld_lineage"
            evidence = item.proxy_evidence
            sources = [SourceExcerpt(reference=ref, statement=catalog[ref].value,
                qualifications=catalog[ref].qualifications, conditions=getattr(catalog[ref], "conditions", ""),
                timing=getattr(catalog[ref], "timing", ""), basis=catalog[ref].basis, status=catalog[ref].status)
                for ref in evidence.references if ref in catalog]
            evidence.basis = "deduced"
            evidence.admission = ClaimAdmission(
                status="admitted", authority="deduction", rule=rule, predicate="diagnostic_lineage",
                source_refs=evidence.references, sources=sources,
                scope=ClaimScope(variable_ids=item.variable.column_ids, effect_ids=item.effect_ids,
                    treatment_ids=item.treatment_ids, assignment_event_key=item.assignment_event_key),
                original_proposal={}, reason="Harness-computed finite diagnostic access disposition over the frozen lineage and admitted timing. Unknown or qualified lineage remains withheld; no causal absence is inferred.",
            )
    return value
