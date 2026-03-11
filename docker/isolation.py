"""Linux kernel restrictions applied inside the Docker worker, before untrusted Python.

No best-effort mode: chroot, dropped credentials and seccomp must enforce the policy.
The launcher has no application credentials, network, Docker API or host mounts.
"""

import ctypes as C
import errno
import os
import resource
from pathlib import Path


def checked(value):
    if value < 0:
        raise OSError(C.get_errno(), os.strerror(C.get_errno()))
    return value


def restrict(work: Path, inputs: Path):
    libc = C.CDLL(None, use_errno=True)
    checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
    # The image contains a library-only root. The queue and supervisor are outside
    # it; /proc is absent. No outside directory descriptor survives exec.
    os.chroot("/runtime")
    os.chdir("/")

    # No sockets, process escape, tracing or inter-process memory access. Docker's
    # default seccomp remains in force as well. Resolve syscall IDs per architecture.
    sec = C.CDLL("libseccomp.so.2", use_errno=True)
    sec.seccomp_init.argtypes, sec.seccomp_init.restype = [C.c_uint32], C.c_void_p
    sec.seccomp_rule_add.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint]
    sec.seccomp_load.argtypes = [C.c_void_p]
    sec.seccomp_release.argtypes = [C.c_void_p]
    ctx = sec.seccomp_init(0x7FFF0000)
    if not ctx:
        raise RuntimeError("seccomp initialization failed")
    try:
        for name in ("socket", "socketpair", "setsid", "setpgid", "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd", "io_uring_setup"):
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError("seccomp rule failed: " + name)
        if sec.seccomp_load(ctx) != 0:
            raise RuntimeError("seccomp enforcement failed")
    finally:
        sec.seccomp_release(ctx)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (5 * 1024 * 1024, 5 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    os.setgroups([])
    os.setgid(20000)
    os.setuid(20000)
    checked(libc.prctl(4, 0, 0, 0, 0))  # PR_SET_DUMPABLE


if __name__ == "__main__":
    work, inputs = Path("/task/work"), Path("/task/input")
    restrict(work, inputs)
    os.chdir(work)
    os.execve(
        "/usr/local/bin/python",
        ["python", "-I", str(work / "code.py")],
        {
            "PATH": "/usr/local/bin:/usr/bin",
            "HOME": str(work),
            "TMPDIR": str(work / "tmp"),
            "DATA_CSV": str(inputs / "table.csv"),
            "INPUTS_DIR": str(inputs / "bundle"),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(work / "tmp"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )
