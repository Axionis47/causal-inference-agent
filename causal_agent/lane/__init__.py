"""The harness every lane runs on: what is shared is shape, not method.

    intake    the table the lane works on: the scope's filter and window applied by code, every pack-named column loaded
    case      the pack weighed by code: facts, open fields, and the flags a lane's beliefs.yaml declares
    verify    a model answer against the pack's facts: a contradiction needs a cite
    asks      one question back to the desk, the same shape for every lane
    records   the artifacts, the result, the report tail: declines and asks in every lane's record

Each engine stays in its family's lane under causal_agent/families/<family>/lane. Nothing here fits a model."""
