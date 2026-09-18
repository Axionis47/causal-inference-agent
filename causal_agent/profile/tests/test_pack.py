from causal_agent.profile.datasets import dataset_entries, load_dataset_pack


def test_all_packs_load_with_every_column_noted():
    for name in dataset_entries():
        pack = load_dataset_pack(name)
        assert pack.unnoted_columns == [], (name, pack.unnoted_columns)
        assert pack.unprofiled_notes == [], (name, pack.unprofiled_notes)
        assert pack.dataset.note
        assert len(pack.changes) >= 1


def test_students_addresses_and_index():
    pack = load_dataset_pack("students")
    assert pack.resolve("col:test_preparation_course.note")
    assert pack.resolve("col:lunch.profile.varies_over")
    assert pack.resolve("change:1.note")
    assert pack.resolve("dataset.profile.grain")
    assert not pack.resolve("col:nope.note")
    index = pack.column_index()
    assert index.count("\n") + 1 == len(pack.columns)
    card = pack.column("test preparation course")
    assert card is not None and "depended on lunch" in (card.note or "")
    assert "[col:test_preparation_course.profile.top_values]" in card.render()
