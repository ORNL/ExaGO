"""Tie the visualization server's lifetime to that of the launcher."""

import os
import signal
import subprocess
import sys
import threading


def start_viz_server(command, *, cwd, env):
    """Start a supervisor with a pipe held open by the ExaGO launcher."""
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *command],
        stdin=subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )


def supervise(command):
    stopped = threading.Event()

    def watch_launcher():
        while os.read(sys.stdin.fileno(), 1):
            pass
        stopped.set()

    def stop(signum, frame):
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server = subprocess.Popen(command, start_new_session=True)
    threading.Thread(target=watch_launcher, daemon=True).start()
    try:
        while server.poll() is None and not stopped.wait(0.2):
            pass
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait()


if __name__ == "__main__":
    supervise(sys.argv[1:])
