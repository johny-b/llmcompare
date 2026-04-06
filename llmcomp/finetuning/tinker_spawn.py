"""Spawn ``tinker_worker`` in a new session and reparent it under init (double-fork).

When the parent is a Jupyter / Cursor notebook kernel, restarting the kernel often
terminates direct child processes. After this dance the training process is no longer
a descendant of the kernel, so it keeps running.

On platforms without ``os.fork`` (Windows), we ``exec`` the worker in the subprocess
without detaching — same behavior as before.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m llmcomp.finetuning.tinker_spawn <run_dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = sys.argv[1]
    argv = [sys.executable, "-m", "llmcomp.finetuning.tinker_worker", run_dir]
    env = os.environ

    if not hasattr(os, "fork"):
        os.execvpe(sys.executable, argv, env)

    pid = os.fork()
    if pid > 0:
        os._exit(0)

    os.setsid()
    pid = os.fork()
    if pid > 0:
        os._exit(0)

    os.execvpe(sys.executable, argv, env)


if __name__ == "__main__":
    main()
