# 7. A picture is drawn on request; no figure is drawn before the run

Decided 2026-09-28, folded into ADR 0006 at the time and recorded here on its own.

Before this, each family registered figures the desk drew at the ready moment to make the family's point, an overlap chart for
adjustment among them, and a figure-pick graph chose which to show. That was a second reasoning path with its own registry, and
it drew what the family wanted shown rather than what the person asked to see.

Now a picture exists only because the person asked for one. The Reader or the Explainer reads the request out of the message as
a judgement; the desk hands it to the drawing tool with what is known about the file; the tool has a model write one script,
runs it in a sandbox, and keeps the code, the picture and every number it shows as facts with addresses. A picture settles no
field. It is stored under the memory before any run and under the design after one, so it is deleted with them, and it is never
read back by code. The chat cites `artifact:<id>` and `artifact:<id>.<fact>` and the gate checks the numbers like any other.

The figures a lane draws from its own artifacts after a run stay. They are data, not images, they are checked against the run's
addresses, and they are the run's evidence rather than a pitch for the design.

Consequences: the pre-run figure registry, the figure-pick graph and the per-family pre-run figure modules are gone; the drawing
tool (`viz/draw.py`, `viz/sandbox.py`, `viz/store.py`) and the `draw` and `draw_after` nodes replace them; the sandbox is a
subprocess by default and a container when `VIZ_SANDBOX=docker`; the demo screenshots that show a ready-moment figure are stale
until re-recorded.
