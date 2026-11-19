from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
#TEMP SCRIPT WHILE WE KEEP STREAMLIT AND Service Files
def stop_process(process: subprocess.Popen[object] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        # Ctrl+C is delivered to the console children as well. Give them a
        # moment to perform their own graceful shutdown before forcing them.
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main() -> int:
    environment = os.environ.copy()
    environment.setdefault("API_SERVER_URL", "http://localhost:8090")

    api_server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "services.api.server:app", "--host", "0.0.0.0", "--port", "8090"],
        cwd=ROOT,
        env=environment,
    )
    streamlit = None
    try:
        print(f"API server: {environment['API_SERVER_URL']}", flush=True)
        streamlit = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "main.py"],
            cwd=ROOT,
            env=environment,
        )
        return streamlit.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        stop_process(streamlit)
        stop_process(api_server)


if __name__ == "__main__":
    raise SystemExit(main())
