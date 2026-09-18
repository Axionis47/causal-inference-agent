from causal_agent.profile import pack
from causal_agent.server.context import render_context


def test_context_renders_the_three_headings_the_pack_loader_reads():
    text = render_context("Students", "Each row is one student.  Every student who sat is included", "A prep course before the exam",
                          [("math score", "the exam mark"), ("lunch", "")])
    assert text == (
        "# Students\n\n"
        "## About the dataset\nEach row is one student. Every student who sat is included.\n\n"
        "## What changed\nA prep course before the exam.\n\n"
        "## About each column\n**math score** — the exam mark.\n\n**lunch** — Not described.\n"
    )
    sections = pack._split_sections(text)
    assert set(sections) == {"about the dataset", "what changed", "about each column"}
    cols = pack._parse_columns(sections["about each column"])
    assert cols["math_score"][0] == "the exam mark." and cols["lunch"][0] == "Not described."
    assert pack._parse_changes(sections["what changed"])[0].note == "A prep course before the exam."
