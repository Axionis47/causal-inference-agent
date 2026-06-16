"""Source for the notebook's figures, built inline from the recomputed data.

matplotlib with the Agg backend so they render headless under nbclient; no
seaborn, to avoid an extra dependency in a user-facing executable. Each figure
is saved to a buffer and shown with ``display(Image(...))`` so it is captured
as a cell output regardless of the kernel's default backend. Every figure
depicts the same result the app shows the user, drawn from the objects the
recompute cells bind, never from a separate re-analysis.
"""
from __future__ import annotations

# A coefficient/CI forest plot of every estimand in `effects` (from ESTIMATE).
FOREST = """\
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display

rows = effects.dropna(subset=['estimate']).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(6, 0.6 * len(rows) + 1))
y = list(range(len(rows)))
ax.errorbar(
    rows['estimate'], y,
    xerr=[
        (rows['estimate'] - rows['ci_lower'].fillna(rows['estimate'])).clip(lower=0),
        (rows['ci_upper'].fillna(rows['estimate']) - rows['estimate']).clip(lower=0),
    ],
    fmt='o', color='#4f46e5', ecolor='#9ca3af', capsize=4,
)
ax.axvline(0, ls='--', lw=1, color='#9ca3af')
ax.set_yticks(y)
ax.set_yticklabels(rows['estimand'])
ax.set_xlabel('effect (95% CI)')
ax.set_title('Estimated effects')
buf = io.BytesIO()
fig.savefig(buf, format='png', bbox_inches='tight')
plt.close(fig)
display(Image(data=buf.getvalue()))"""
