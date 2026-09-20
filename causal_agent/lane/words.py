"""What a check asks, in the question's words: declared per lane under `in_words` in its checks.yaml, put in front of
the number on every check line the interpretation and the brief read. A check with no sentence keeps its name."""

from __future__ import annotations

from causal_agent.common.contracts import CheckResult

FROM_THE_PERSON = ("belief.", "unknown.", "contradiction.")


def check_words(cfg: dict, name: str, names: dict[str, str] | None = None) -> str:
    """The sentence for a check name; a per-column name such as balance.lunch matches its prefix and fills {column}."""
    table = cfg.get("in_words") or {}
    names = names or {}
    if name in table:
        return str(table[name])
    head, _, col = name.partition(".")
    if col and head in table:
        return str(table[head]).replace("{column}", names.get(col, col))
    return name


def say(results: list[CheckResult], cfg: dict, names: dict[str, str] | None = None) -> list[CheckResult]:
    """Each data check's detail led by its sentence; a flag from what the person said already carries its own."""
    for r in results:
        if r.name.startswith(FROM_THE_PERSON):
            continue
        w = check_words(cfg, r.name, names)
        if w != r.name and not r.detail.startswith(w):
            r.detail = f"{w}: {r.detail}" if r.detail else w
    return results
