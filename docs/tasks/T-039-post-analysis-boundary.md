# T-039 — Post-analysis boundary

Status: implementation complete and functionally verified. Complexity and separate
product-release gates remain open. Earlier planning-only restrictions were
superseded by the user's explicit go-ahead.

The user also authorized full study prompts, outputs, tools and provider-exposed
reasoning summaries in LangSmith. Credentials remain redacted; unavailable model
internal reasoning cannot be captured. Ordinary operational logging remains enabled.

The canonical [post-analysis documents](../post_analysis/README.md) describe the
contract, actual graph, input coverage, removal map and compatibility boundaries.
The package [AGENTS.md](../../src/causal/post_analysis/AGENTS.md) governs ownership,
deletion, invariant checks, trace coverage and future graph maintenance.

Implementation consolidates the live reporting path under one package, removes the
old claim and curator agents, emits a numerical-only handoff, and introduces an
LLM tool loop with independent image-aware review and exact report release.
The [verification record](T-039-verification.md) includes the full suite, final
targeted checks, lint/type checks, packaged resources, measured budget breaches
and compatibility limits. No live model or LangSmith execution was performed.
