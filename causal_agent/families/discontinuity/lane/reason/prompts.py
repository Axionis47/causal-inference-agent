"""Bounded RD judgements; observations never certify identifying assumptions."""

from causal_agent.explore.diagnostics import DIAGNOSTIC_RULE

CITE = "Copy the exact bracketed citation address, including its claim:, col:, ladder:, check: or probe: prefix. Cite only addresses shown to you or returned by your tools. Keep reported study facts, causal assumptions and statistical diagnostics distinct."
COVARIATES = (
    """Classify every listed column once. Predetermined means it describes something fixed before assignment; a later measurement of an earlier fixed characteristic can qualify when its account says so. Exclude changed-by-treatment columns, outcome measures, and identifiers. Give a cited reason for each positive claim. Preserve confirmed timing. A balance test cannot rewrite when a variable was measured. Unknown roles can be retained as uncertainty; never claim missing evidence was supplied. """
    + CITE
)
LINE = (
    """Assess whether this cutoff comparison has a defensible continuity account, conditional on its stated assumptions. Explain manipulation, coincident interventions and score timing risks. Density and covariate continuity tests can contest the account, but passing tests cannot prove it. Failed tests do not prove manipulation; retain the confirmed mechanism and describe a soft conflicting diagnostic conditionally. Cite every flagged diagnostic you answer. A genuine coincident intervention or score set after assignment blocks the requested interpretation. Known score movability plus bunching requires an explicit manipulation risk. """
    + CITE
    + "\n\n"
    + DIAGNOSTIC_RULE
    + "\nDiagnostic coding is optional when the existing evidence settles this decision. Use it for an unresolved score-support, "
    "heaping, missingness, predetermined-covariate balance or receipt question. DATA_CSV contains oriented score x, exact admitted "
    "eligibility side, optional receipt t and cluster, and only approved predetermined covariates under cov_ aliases. Read the "
    "input mapping and equality rule shown in the view. No target outcome or other outcome proxy is available. Generated "
    "diagnostics cannot establish continuity or local random assignment, override a hard check, select a target, or change "
    "estimator, inference, bandwidth selector or window policy. The engine retains those deterministic policies. "
    "Once code has been attempted, put at least one successful numerical probe address in Line.cites, including when its "
    "finding leaves the identifying uncertainty unresolved. Explain its limitation rather than silently ignoring it."
)
ASSUMPTIONS = (
    """Evaluate every offered semantic endpoint once. Admissible means that its own target is identifiable conditional on the admitted premises, not that it answers the exact wording of the human question. Eligibility ITT endpoints are deliberately included as companion designs when receipt is asked about: evaluate their distinct target rather than rejecting them for being different from receipt. State the assumptions on which it can proceed, referencing the study account rather than treating diagnostics as proof. The absence of predetermined covariates or a noncomputable density test is not a prerequisite failure: these are optional falsification screens, not proof of the identifying assumptions. Positive admitted human assumptions remain explicitly conditional premises. Do not reject solely because such proof is unavailable. A failed diagnostic also cannot prove manipulation or erase a confirmed exogenous assignment account. In an explicitly independently generated study, carry a soft conflicting diagnostic as a caveat rather than converting a false-positive test into a factual mechanism. A cutoff effect and an effect inside a fixed window differ in population. Receipt and eligibility differ in contrast. Local randomisation needs an admitted as-if-random account; a discrete score alone is insufficient. Fuzzy receipt needs exclusion and monotonicity assumptions. Missing human facts cannot be invented. Reject unsupported endpoints with a reason. """
    + CITE
)
