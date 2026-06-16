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


# The causal DAG laid out like the app: confounders left (latent on top),
# treatment centre, outcome right; the adjustment set is ringed indigo, latent
# nodes are hollow with a rose edge, and the treatment->outcome edge is bold.
DAG_FIGURE = """\
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from IPython.display import Image, display
from src.analysis_v2.spec import CausalDAG

dag = CausalDAG.model_validate(DAG)
adj = dag.adjustment_set()
latent = {n.name for n in dag.nodes if not n.observed}
treat, out = dag.treatment, dag.outcome
confs = [n.name for n in dag.nodes if n.name not in (treat, out)]
ordered = [c for c in confs if c in latent] + [c for c in confs if c not in latent]
pos = {}
if treat is not None:
    pos[treat] = (1.0, 0.5)
if out is not None:
    pos[out] = (2.0, 0.5)
for i, name in enumerate(ordered):
    pos[name] = (0.0, 1.0 - (i + 1) / (len(ordered) + 1))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axis('off')
ax.set_xlim(-0.7, 2.7)
ax.set_ylim(-0.05, 1.05)
for e in dag.edges:
    a, b = pos.get(e.source), pos.get(e.target)
    if a is None or b is None:
        continue
    spine = (e.source == treat and e.target == out)
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle='-|>', mutation_scale=12,
        lw=1.6 if spine else 1.0, alpha=1.0 if spine else 0.3,
        color='#6b7280', shrinkA=13, shrinkB=13, zorder=1,
    ))
for name, (x, yy) in pos.items():
    is_lat = name in latent
    if name == treat:
        face = '#f59e0b'
    elif name == out:
        face = '#10b981'
    else:
        face = 'none' if is_lat else '#e5e7eb'
    edge = '#f43f5e' if is_lat else '#374151'
    if name in adj:
        ax.scatter([x], [yy], s=1000, facecolors='none',
                   edgecolors='#4f46e5', linewidths=1.3, zorder=2)
    ax.scatter([x], [yy], s=600, facecolors=face, edgecolors=edge,
               linewidths=1.3, zorder=3)
    ax.annotate(name, (x, yy), xytext=(0, -18), textcoords='offset points',
                ha='center', fontsize=8)
ident, _reason = dag.is_identifiable()
ax.set_title(
    f"Causal model | adjustment set: {sorted(adj) or 'empty'} | "
    f"identified by adjustment: {ident}",
    fontsize=9,
)
buf = io.BytesIO()
fig.savefig(buf, format='png', bbox_inches='tight')
plt.close(fig)
display(Image(data=buf.getvalue()))
print('suspected latent confounding present:', dag.has_latent_confounding())"""


# A love plot of |standardized mean difference| per covariate (from BALANCE),
# with the 0.1 rule-of-thumb line; shows after-matching too when available.
LOVE = """\
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display

if len(BALANCE):
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(BALANCE) + 1.2))
    ax.scatter(BALANCE['smd_before'].abs(), BALANCE['covariate'],
               color='#9ca3af', label='before', zorder=3)
    if 'smd_after' in BALANCE.columns:
        ax.scatter(BALANCE['smd_after'].abs(), BALANCE['covariate'],
                   color='#4f46e5', label='after matching', zorder=3)
        ax.legend(loc='lower right', fontsize=8)
    ax.axvline(0.1, ls='--', lw=1, color='#f43f5e')
    ax.set_xlabel('|standardized mean difference|')
    ax.set_title('Covariate balance')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    display(Image(data=buf.getvalue()))
else:
    print('no covariates to assess balance for this design')"""
