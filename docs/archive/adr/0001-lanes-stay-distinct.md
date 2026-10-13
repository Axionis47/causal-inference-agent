# 1. Each family keeps its own engine and design

Adjustment runs on DoWhy, diff-in-diff on pyfixest, discontinuity on rdrobust. A lane exists because its design is its
own: what it assumes, what it can say, what it must not guess. Folding one engine under all three would blur that.

What is shared is the harness shape only: the pack weighed by code, the model asked only what the pack leaves open, an
ask-back, recorded declines, a figure tail, and the plumbing every lane copied (`causal_agent/lane/`). A lane is
improved by giving it more of that shape, never by moving another lane's method into it.
