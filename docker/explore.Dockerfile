# The exploration agent's container: Python with pandas, numpy, scipy, statsmodels and matplotlib, nothing else. Built once
# with `make explore-image`; used when EXPLORE_SANDBOX=docker. A script runs as the caller's user, with no network, the table
# mounted read-only and the step's folder mounted as its working directory.
FROM python:3.12-slim
RUN pip install --no-cache-dir "pandas>=3,<4" "numpy>=2.4,<3" "scipy>=1.15,<2" "statsmodels>=0.15,<1" "matplotlib>=3.11,<4" && useradd -m explore
USER explore
WORKDIR /work
