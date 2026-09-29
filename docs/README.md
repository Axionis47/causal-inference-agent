# Reading about it

One page per mechanism. Each opens with a picture, walks one real example, and links into the code and to the decision it rests on.

| page | what it answers |
|---|---|
| [architecture.md](architecture.md) | how the code is laid out: the components, the layers, a family package, the stores, the wire, the checks |
| [desk.md](desk.md) | the one graph: the question first, one question per turn, the run, the chat after; the three interrupts; the routing as code |
| [memory-and-matrix.md](memory-and-matrix.md) | a field with a status and a source; the one write path; the matrix that turns claims into a design choice |
| [gates.md](gates.md) | every model call in the system, what it is given, what it returns, what code checks, what happens on failure |
| [pack-and-addresses.md](pack-and-addresses.md) | the only input a lane sees, and the address grammar every cite is checked against |
| [lanes.md](lanes.md) | the harness, the adjustment lane stage by stage, the other two lanes |
| [drawing-tool.md](drawing-tool.md) | how a picture is asked for, drawn in a sandbox, stored, and cited |
| [testing.md](testing.md) | how every judgement is scripted, where a test writes, the evals |
| [page.md](page.md) | the routes, the one view, the generated wire, the inspector |
| [demo.md](demo.md) | one real run walked, and the records it left under `demo/` |
| [diagrams/](diagrams/README.md) | the pictures and the visual language they share |
| [adr/](adr/) | the decisions the layout rests on |
