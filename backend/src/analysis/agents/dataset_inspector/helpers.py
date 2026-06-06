"""Pure scoring functions used by dataset_inspector to pick the winner.

The agent fans out a full data_profiler invocation per candidate file
and collects the resulting DataProfile objects. This module turns that
collection into a single choice + a one-line reason — deterministically,
no LLM, no agent state. Same inputs always produce the same winner.

The rubric is the contract: changing weights here changes which file
gets picked across every multi-file Kaggle bundle the system has ever
seen, so every weight has a test pinning it.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.analysis.agents.data_profiler import DataProfile

# Rubric weights. Sourced from the plan; pinned by tests/test_helpers.py.
SCORE_HAS_TREATMENT = 3.0
SCORE_HAS_OUTCOME = 3.0
SCORE_PER_CONFOUNDER = 1.0
SCORE_CONFOUNDER_CAP = 5.0
SCORE_TRAIN_PATTERN = 2.0
SCORE_TEST_PATTERN = -2.0
SCORE_TINY_FILE = -5.0
TINY_FILE_THRESHOLD = 50  # rows; below this causal estimation is unstable

# Keyword patterns matched against the lowercased Kaggle resource
# description (or the filename if no description is available). Train-
# like files are usually the analysis target; test / submission files
# are the wrong target by definition (no labels, no treatment column).
TRAIN_PATTERNS = ("train", "training", "main data")
TEST_PATTERNS = ("test", "holdout", "sample", "submission", "example")


@dataclass(frozen=True)
class FileScore:
    """One candidate file's score, ready for ranking + explanation.

    Frozen so accidentally mutating a score post-rank is impossible. The
    reason list is built up as each rubric rule fires so the winner can
    quote the reasons that actually applied to it.
    """

    name: str
    score: float
    reasons: tuple[str, ...]
    n_samples: int


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in patterns)


def score_profile(
    name: str,
    profile: DataProfile,
    *,
    resource_description: str | None = None,
) -> FileScore:
    """Score a single candidate file via the fixed rubric.

    `resource_description` is the Kaggle `dataset-metadata.json` resource
    description if available (e.g. "Training set with 14 features").
    When absent, train / test patterns are matched against the filename
    instead — weaker signal but still useful.
    """
    score = 0.0
    reasons: list[str] = []

    if profile.treatment_candidates:
        score += SCORE_HAS_TREATMENT
        reasons.append(
            f"has {len(profile.treatment_candidates)} treatment candidate"
            f"{'s' if len(profile.treatment_candidates) != 1 else ''}"
        )
    if profile.outcome_candidates:
        score += SCORE_HAS_OUTCOME
        reasons.append(
            f"has {len(profile.outcome_candidates)} outcome candidate"
            f"{'s' if len(profile.outcome_candidates) != 1 else ''}"
        )
    if profile.potential_confounders:
        bonus = min(
            len(profile.potential_confounders) * SCORE_PER_CONFOUNDER,
            SCORE_CONFOUNDER_CAP,
        )
        score += bonus
        reasons.append(f"{len(profile.potential_confounders)} potential confounders (+{bonus:g})")

    haystack = resource_description if resource_description else name
    if _matches_any(haystack, TRAIN_PATTERNS):
        score += SCORE_TRAIN_PATTERN
        reasons.append("matches train-set pattern")
    elif _matches_any(haystack, TEST_PATTERNS):
        score += SCORE_TEST_PATTERN
        reasons.append("matches test/sample/submission pattern")

    if profile.n_samples < TINY_FILE_THRESHOLD:
        score += SCORE_TINY_FILE
        reasons.append(
            f"only {profile.n_samples} rows (< {TINY_FILE_THRESHOLD} threshold for causal estimation)"
        )

    return FileScore(
        name=name,
        score=score,
        reasons=tuple(reasons),
        n_samples=profile.n_samples,
    )


def pick_winner(scores: list[FileScore]) -> FileScore:
    """Return the highest-scored file; ties broken by larger n_samples.

    Raises ValueError on empty input — caller is expected to handle the
    no-files case at the preflight stage, not here.
    """
    if not scores:
        raise ValueError("pick_winner requires at least one FileScore")
    # Highest score first; tie-break: larger n_samples; then lexical name
    # so the order is fully deterministic even on identical fixtures.
    return max(scores, key=lambda s: (s.score, s.n_samples, s.name))


def explain_choice(winner: FileScore, scores: list[FileScore]) -> str:
    """One-line human-readable reason the winner won.

    Quoted in the AnalysisDecision the agent pushes and shown in the UI
    next to the selected file. Mentions the winner's score, the
    runner-up's score, and the reasons that contributed to the win.
    """
    runner_up = None
    for s in sorted(scores, key=lambda s: (-s.score, -s.n_samples, s.name)):
        if s.name != winner.name:
            runner_up = s
            break

    reason_clause = "; ".join(winner.reasons) if winner.reasons else "no rubric rules matched"
    head = f"{winner.name} (score {winner.score:g}): {reason_clause}"

    if runner_up is not None:
        head += f"; next: {runner_up.name} score {runner_up.score:g}"
    return head
