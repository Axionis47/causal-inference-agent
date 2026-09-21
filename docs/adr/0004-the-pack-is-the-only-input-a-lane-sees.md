# 4. The pack is the only input a lane sees

A lane reads the Handoff and the CSV, nothing else: no memory, no note, no dataset name in its knowledge. It runs in
its own process on a hand-off file, and what it writes back is addressed artifacts the desk can cite. Where the lane does
not take the pack as given it records a Decline with the value it took instead, so the desk can show where it disagreed.

The stored hand-offs under each family's `evals/handoffs/` are that contract frozen: a routing change cannot mask a lane
regression.
