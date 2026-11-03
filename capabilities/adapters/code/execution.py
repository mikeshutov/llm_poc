from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent_runtime.context import get_current_conversation_id, get_current_user_id
from common.config import ANONYMOUS_OWNER
from common.config import FILES_DIR
from capabilities.registry.constants import (
    TOOL_NAME_CODE_EXECUTE,
    TOOL_RESULT_TYPE_CODE_EXECUTION,
)
from retrieval.evidence import EvidenceView, ToolResult


DEFAULT_IMAGE = "python:3.12-slim"
MAX_CODE_LENGTH = 100_000
MAX_TIMEOUT_SECONDS = 60
MAX_OUTPUT_LENGTH = 20_000


class CodeExecuteArgs(BaseModel):
    code: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CODE_LENGTH,
        description="Python code to execute in the isolated sandbox.",
    )
    timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=MAX_TIMEOUT_SECONDS,
        description="Maximum execution time in seconds.",
    )


class CodeExecutionResult(BaseModel):
    language: Literal["python"] = "python"
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    sandbox: str = "docker"
    error: str = ""


def _truncate(value: str | bytes) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) <= MAX_OUTPUT_LENGTH:
        return value
    return value[:MAX_OUTPUT_LENGTH] + "\n[output truncated]"


def _docker_workspace_path(path: Path) -> str:
    # Docker Desktop accepts forward-slash bind paths on Windows.
    return str(path.resolve()).replace("\\", "/")


def _conversation_workspace() -> Path | None:
    user_id = (get_current_user_id() or ANONYMOUS_OWNER).strip()
    conversation_id = (get_current_conversation_id() or "").strip()
    if (
        not user_id
        or user_id in {".", ".."}
        or any(char in user_id for char in "/\\")
        or not conversation_id
        or conversation_id in {".", ".."}
        or any(char in conversation_id for char in "/\\")
    ):
        return None
    workspace = (Path(FILES_DIR) / user_id / conversation_id).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _run_python_in_docker(
    code: str,
    timeout_seconds: int,
    workspace_path: Path | None = None,
) -> CodeExecutionResult:
    image = os.getenv("CODE_EXECUTION_DOCKER_IMAGE", DEFAULT_IMAGE).strip() or DEFAULT_IMAGE
    with tempfile.TemporaryDirectory(prefix="code-execution-") as sandbox:
        sandbox_path = Path(sandbox)
        script_path = sandbox_path / "main.py"
        script_path.write_text(code, encoding="utf-8")
        execution_workspace = workspace_path or sandbox_path
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "128m",
            "--cpus",
            "0.5",
            "--user",
            "65532:65532",
            "--mount",
            f"type=bind,src={_docker_workspace_path(sandbox_path)},dst=/sandbox,readonly",
            "--mount",
            f"type=bind,src={_docker_workspace_path(execution_workspace)},dst=/workspace",
            "--workdir",
            "/workspace",
            image,
            "python",
            "-I",
            "-S",
            "/sandbox/main.py",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            return CodeExecutionResult(error="Docker is not available to run the code sandbox.")
        except subprocess.TimeoutExpired as exc:
            return CodeExecutionResult(
                stdout=_truncate(exc.stdout or ""),
                stderr=_truncate(exc.stderr or ""),
                timed_out=True,
                error=f"Code execution exceeded the {timeout_seconds}-second timeout.",
            )
        except OSError as exc:
            return CodeExecutionResult(error=f"Unable to start the code sandbox: {exc}")

        return CodeExecutionResult(
            stdout=_truncate(completed.stdout or ""),
            stderr=_truncate(completed.stderr or ""),
            exit_code=completed.returncode,
            error="" if completed.returncode == 0 else "Code execution failed.",
        )


@tool(
    TOOL_NAME_CODE_EXECUTE,
    args_schema=CodeExecuteArgs,
    description=(
        "Execute Python code in an isolated Docker sandbox with no network access, "
        "resource limits, and a conversation-scoped workspace mounted at /workspace. "
        "Files written under /workspace persist for the current conversation. Returns "
        "stdout, stderr, exit code, and timeout status."
    ),
)
def code_execute(code: str, timeout_seconds: int = 10) -> ToolResult:
    result = _run_python_in_docker(code, timeout_seconds, _conversation_workspace())
    summary = (
        result.error
        or f"Python sandbox exited with code {result.exit_code}."
    )
    evidence = EvidenceView(
        item_id=f"python-execution-{result.exit_code}-{result.timed_out}",
        tool_name=TOOL_NAME_CODE_EXECUTE,
        title="Sandboxed Python execution",
        summary=summary,
        source=TOOL_NAME_CODE_EXECUTE,
        entity_type=TOOL_RESULT_TYPE_CODE_EXECUTION,
        llm_metadata=result.model_dump(mode="json"),
    )
    return ToolResult(result=result, evidence=[evidence])


__all__ = [
    "CodeExecutionResult",
    "CodeExecuteArgs",
    "code_execute",
]
