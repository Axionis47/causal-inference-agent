# Pinned Python and analysis libraries for Linux containers on Docker Engine or Docker Desktop.
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
COPY docker/explore.requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements.txt && pip check && useradd -u 10001 -m explore
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libseccomp2 && rm -rf /var/lib/apt/lists/*
# Hard links share the installed library bytes. The child root excludes /queue,
# /proc, /opt/worker and all application paths. /runtime/task is a bounded tmpfs.
RUN mkdir -p /runtime/task /runtime/dev /runtime/etc \
    && cp -al /usr /runtime/usr \
    && cp -a /lib /runtime/lib \
    && if [ -e /lib64 ]; then cp -a /lib64 /runtime/lib64; fi \
    && cp /etc/ld.so.cache /runtime/etc/ld.so.cache
COPY docker/explore_worker.py docker/isolation.py /opt/worker/
USER explore
WORKDIR /work
