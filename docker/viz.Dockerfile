# The drawing tool's container: Python with pandas and matplotlib, nothing else. Built once with `make viz-image`;
# used when VIZ_SANDBOX=docker. The script runs as the caller's user, with no network, the CSV mounted read-only and the
# artifact folder mounted as its working directory.
FROM python:3.12-slim
RUN pip install --no-cache-dir "pandas>=2.2" "matplotlib>=3.9" && useradd -m viz
USER viz
WORKDIR /work
