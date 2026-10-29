def test_meta_is_written_whole_and_leaves_no_temp_file(settings):
    from causal_agent.server import datasets as DS

    DS.write_meta(settings, "x", {"name": "x", "question": None})
    DS.write_meta(settings, "x", {"name": "x", "question": "did it?"})
    assert DS.read_meta(settings, "x") == {"name": "x", "question": "did it?"}
    assert [p.name for p in DS.meta_path(settings, "x").parent.iterdir()] == ["meta.json"]
