def test_number_cannot_be_attached_to_a_nonnumeric_context_line():
    from causal_agent.desk.contracts import AfterReply, NumberStated
    from causal_agent.desk.explainer import gate
    from causal_agent.desk.material import Material

    material = Material()
    material.add("context:meaning", "The outcome measures support.")
    reply = AfterReply(kind="answer", text="Support was 999.", cites=["context:meaning"], numbers=[NumberStated(address="context:meaning", value=999)])
    assert gate(reply, material, "before")
