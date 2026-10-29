"""Phase 3: failures are visible results, not missing or fabricated facts."""

from io import BytesIO

from causal_agent.profile import inspection as P


def fail(*args, **kwargs):
    raise RuntimeError("deliberate calculation failure")


def test_numeric_failure_is_visible(monkeypatch):
    monkeypatch.setattr(P, "numeric_summary", fail)
    result = P.inspect(BytesIO(b"a,b\n1,x\n2,y\n"))
    assert result.status == "partial"
    assert result.columns[0].numeric is None
    assert any(c.name == "numeric" and c.status == "failed" for c in result.columns[0].checks)
    assert result.columns[1].status == "complete"


def test_key_and_column_failure_preserve_structure(monkeypatch):
    monkeypatch.setattr(P, "candidate_keys", fail)
    monkeypatch.setattr(P, "column", fail)
    result = P.inspect(BytesIO(b"a\n1\n"))
    assert result.dataset.rows == 1 and result.status == "partial"
    assert result.columns[0].status == "failed"
    assert result.dataset.checks[-1].status == "failed"
