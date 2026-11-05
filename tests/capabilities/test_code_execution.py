from types import SimpleNamespace

from capabilities.adapters.code.execution import _run_python_in_docker


def test_code_execution_uses_restricted_docker_invocation(monkeypatch) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="4\n", stderr="", returncode=0)

    monkeypatch.setattr("capabilities.adapters.code.execution.subprocess.run", fake_run)

    result = _run_python_in_docker("print(2 + 2)", timeout_seconds=7)

    command = captured["command"]
    assert result.stdout == "4\n"
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "--cap-drop" in command and command[command.index("--cap-drop") + 1] == "ALL"
    assert "--memory" in command and command[command.index("--memory") + 1] == "128m"
    assert "--cpus" in command and command[command.index("--cpus") + 1] == "0.5"
    assert captured["kwargs"]["timeout"] == 7
    assert captured["kwargs"]["check"] is False
