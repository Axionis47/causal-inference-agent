"""Exact Python column references are checked after the outcome has been physically removed."""

import pytest

from causal_agent.explore.gates import names_dropped_column


@pytest.mark.parametrize(
    "code",
    [
        "import numpy as np\nimport matplotlib.pyplot as plt\nprint(np.array([1, 2]).dtype)",
        "yearly = df['age'].mean()\nprint(yearly)",
        "y = [1, 2]\nplt.plot([0, 1], y)",
        "# y is withheld\nprint('outcome y is unavailable')",
        "df.groupby('county')['baseline'].mean()",
        "plt.hist(df['c']); plt.ylabel('Frequency'); plt.grid(axis='y')",
    ],
)
def test_short_outcome_does_not_refuse_ordinary_code(code):
    assert names_dropped_column(code, "y") == ""


@pytest.mark.parametrize(
    "code",
    [
        "print(df['y'].mean())",
        "print(df.y)",
        "df.loc[:, 'y']",
        "df.groupby(['group', 'y'])",
        "df['math score']",
        "df.query('y > 0')",
        "df.eval('y * 2')",
        "df.plot(y='y')",
        "getattr(df, 'y')",
    ],
)
def test_exact_outcome_references_are_refused(code):
    assert "held back" in names_dropped_column(code, "math score" if "math score" in code else "y")


def test_no_outcome_is_withheld_after_freeze():
    assert names_dropped_column("print(df['y'].mean())", None) == ""
