"""The pack's prose: the dataset and the change as the lanes and the routing read them."""

from __future__ import annotations


def render_dataset_text(name: str, facts: dict, grain: dict, sampling: dict, missing: dict, docs: dict | None = None) -> str:
    """The dataset card: what the memory says about the rows, then the profile's facts, every line addressed."""
    docs = docs or {}
    about = []
    if grain.get("row_is"):
        about.append(f"each row is {str(grain['row_is']).rstrip('.')}")
    if grain.get("panel") is True:
        about.append("the same unit appears in more than one period")
    if grain.get("nesting"):
        about.append(f"units sit inside {', '.join(grain['nesting'])}")
    if sampling.get("how"):
        about.append(f"rows were chosen: {sampling['how']}" + (f", {sampling['detail']}" if sampling.get("detail") else ""))
    if missing.get("why"):
        about.append(f"missing values: {missing['why']}")
    lines = [f"[dataset] {name}", f"  [dataset.note] {'. '.join(about) if about else docs.get('about') or '(nothing stated about the rows)'}"]
    lines.append(
        f"  [dataset.profile.rows] {facts.get('rows', '?')} rows, {facts.get('columns', '?')} columns, {facts.get('duplicate_rows', 0)} duplicate rows"
    )
    g = facts.get("grain") or []
    lines.append(f"  [dataset.profile.grain] {' + '.join(g) if g else 'no key column found'}")
    t = facts.get("time_coverage")
    lines.append(
        f"  [dataset.profile.time_coverage] {t['column']}: {t['first']} to {t['last']}, {t.get('inferred_frequency')}, {t.get('gaps')} gaps"
        if t
        else "  [dataset.profile.time_coverage] no time column"
    )
    e = facts.get("entity_summary")
    lines.append(
        f"  [dataset.profile.entity_summary] {' + '.join(e['columns'])}: {e['entities']} entities, {e['rows_per_entity_min']} to {e['rows_per_entity_max']} rows each"
        if e
        else "  [dataset.profile.entity_summary] no entity column declared"
    )
    if facts.get("format_issues"):
        lines.append("  [dataset.profile.format_issues] " + "; ".join(facts["format_issues"]))
    return "\n".join(lines)


def render_change_text(ch: dict, a: dict) -> str:
    """The change card, one paragraph, addressed change:1.note."""
    if not ch.get("what") and not a.get("rule"):
        return "[change:1.note] not stated"
    parts = []
    if ch.get("what"):
        parts.append(
            f"{str(ch['what']).rstrip('.')}." + (f" It reached {ch['to_whom']}," if ch.get("to_whom") else "") + (f" {ch['when']}." if ch.get("when") else "")
        )
    if ch.get("date_column"):
        parts.append(
            f"The period column is {ch['date_column']!r}" + (f", the change took effect at {ch['period_value']!r}." if ch.get("period_value") else ".")
        )
    kind_words = {
        "lottery": "a random draw decided who got it",
        "cutoff_rule": "a line on a measured score decided who got it",
        "own_choice": "units chose whether to take it, with or without an offer",
        "date_by_others": "it reached some units on a date by a decision made above them",
        "third_party": "someone else picked the units one by one",
    }
    if a.get("kind"):
        parts.append(f"How it was assigned: {kind_words.get(a['kind'], a['kind'])}.")
    if a.get("rule"):
        parts.append(f"The rule, in the person's words: {str(a['rule']).rstrip('.')}.")
    if a.get("depends_on"):
        parts.append(f"The decision or the offer depended on {', '.join(a['depends_on'])}.")
    if a.get("treatment_column"):
        parts.append(
            f"{a['treatment_column']!r} records who got it" + (f", {a['treated_level']!r} meaning yes." if a.get("treated_level") is not None else ".")
        )
    if a.get("score_column"):
        side = a.get("treated_side") or "above"
        parts.append(
            f"The score is {a['score_column']!r}, treated {side} {a.get('cutoff')}"
            + (", the cutoff value itself treated." if a.get("cutoff_value_treated") else ".")
        )
    if a.get("level_column"):
        parts.append(f"It was assigned at the level of {a['level_column']!r}.")
    if a.get("movable") is not None:
        parts.append("A unit could change what the rule looked at." if a["movable"] else "A unit could not change what the rule looked at.")
    return "[change:1.note] " + " ".join(parts)
