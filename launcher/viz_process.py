"""Tie the visualization server's lifetime to that of the launcher."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def warm_viz_cache(
    process, base_url, instance, *, filename="opflowout.json", timeout=300
):
    """Prepare the selected case requested by the user."""
    deadline = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                "Visualization server exited. Check its logs and whether port 8085 is in use."
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "Timed out waiting for the visualization server. Check its logs and retry preparation."
            )
        try:
            with opener.open(
                f"{base_url}/health", timeout=min(1, remaining)
            ) as response:
                health = json.load(response)
        except (OSError, URLError, ValueError):
            time.sleep(min(0.2, remaining))
            continue
        if health.get("instance") != instance:
            raise RuntimeError(
                "A different visualization server is using port 8085. Stop it before launching this server."
            )
        break

    request = Request(
        f"{base_url}/cache/warm",
        data=json.dumps({"filename": filename}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "Timed out waiting for visualization preparation. Retry preparation to check its result."
            )
        with opener.open(request, timeout=remaining) as response:
            result = json.load(response)
    except HTTPError as error:
        try:
            failure = json.load(error)
        except ValueError:
            raise RuntimeError(
                f"Visualization preparation failed (HTTP {error.code}). Check the server logs."
            ) from error
        if failure.get("map") == "ready":
            message = f"Map prepared, but ChatGrid preparation failed: {failure['chatgrid']['error']}"
        else:
            message = failure.get("error", "Visualization preparation failed.")
        raise RuntimeError(message) from error
    except (OSError, URLError) as error:
        if process.poll() is not None:
            raise RuntimeError(
                "Visualization server exited during preparation. Check its logs."
            ) from error
        raise RuntimeError(
            "Visualization preparation did not finish within the wait period. Retry preparation to check its result."
        ) from error
    if not result.get("ready"):
        raise RuntimeError(
            "Visualization preparation is incomplete. Check the server logs."
        )
    return result


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
