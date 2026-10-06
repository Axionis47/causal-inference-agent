"""The exploration agent: a read-only look at the data on request, from any turn of the conversation. One tool, `run_code`,
runs a script in the sandbox and hands back what it printed and wrote; a bounded episode (`lane/episode.py`) decides what to
run next within a budget; one typed record names the step whose files are the answer, the facts it stands behind, the picture
and the table. Code gates the record, and before a design has run the table it is sent has the outcome or the treatment taken
out, so it cannot estimate the effect. What it made lands on the artifacts shelf (`memory/artifacts.py`)."""
