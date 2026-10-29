"""Phase 3: dataset measurements are distinct from row meaning."""

from io import BytesIO

from causal_agent.profile import inspection as P


def test_structure_quotes_bom_and_duplicate_rows():
    result = P.inspect(BytesIO(b'\xef\xbb\xbfperson,visit,note\r\na,1,"hello,\nworld"\r\na,2,x\r\nb,1,y\r\nb,1,y\r\n'))
    assert result.dataset.rows == 4 and result.dataset.columns == 3 and result.dataset.duplicate_rows == 1
    assert result.dataset.candidate_keys == []
    assert result.columns[2].top_values[1].value == "hello,\nworld"


def test_keys_exclude_empty_fields_and_include_pairs():
    assert P.candidate_keys([["a", "1", ""], ["a", "2", "x"], ["b", "1", "y"], ["b", "2", "z"]], 3) == [[0, 1]]


def test_keys_have_no_three_column_search():
    rows = [[str((n >> i) % 2) for i in range(3)] for n in range(8)]
    assert P.candidate_keys(rows, 3) == []
    assert "No larger combinations" in P.inspect(BytesIO(b"a,b\n1,x\n2,y\n")).dataset.key_search_limit


def test_deterministic_and_no_input_rewrite():
    data = b"id,x\n001,NA\n002,unknown\n"
    stream = BytesIO(data)
    assert P.inspect(stream) == P.inspect(BytesIO(data))
    assert stream.getvalue() == data
