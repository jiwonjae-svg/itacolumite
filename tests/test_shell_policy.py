"""Tests for the action policy / shell request classifier."""

import pytest

from itacolumite.action.shell import RiskLevel, ShellRequest, classify_request


def _req(program: str, args: list[str] | None = None) -> ShellRequest:
    return ShellRequest(program=program, args=args or [])


class TestClassifyRequest:
    """Test shell request risk classification."""

    # ── Safe programs ────────────────────────────────────────

    @pytest.mark.parametrize("program,args", [
        ("git", ["status"]),
        ("git", ["log", "--oneline"]),
        ("git", ["diff"]),
        ("python", ["--version"]),
        ("node", ["--version"]),
        ("where.exe", ["python"]),
        ("whoami", []),
        ("Test-Path", ["C:\\temp"]),
        ("test-path", ["C:\\temp"]),
        ("Get-ChildItem", ["C:\\temp"]),
        ("dir", ["C:\\temp"]),
        ("ls", ["C:\\temp"]),
        ("mkdir", ["C:\\temp\\demo"]),
        ("md", ["C:\\temp\\demo"]),
    ])
    def test_safe_requests(self, program: str, args: list[str]) -> None:
        assert classify_request(_req(program, args)) == RiskLevel.SAFE

    # ── Normal programs (build/test/install) ─────────────────

    @pytest.mark.parametrize("program,args", [
        ("pytest", ["-v"]),
        ("npm", ["test"]),
        ("npm", ["run", "build"]),
        ("pip", ["install", "flask"]),
        ("npm", ["install"]),
        ("code", ["C:\\temp\\demo"]),
        ("notepad.exe", []),
        ("Start-Process", ["notepad.exe"]),
        ("git", ["add", "-A"]),
        ("git", ["fetch", "origin"]),
    ])
    def test_normal_requests(self, program: str, args: list[str]) -> None:
        assert classify_request(_req(program, args)) == RiskLevel.NORMAL

    # ── Dangerous programs (need approval) ───────────────────

    @pytest.mark.parametrize("program,args", [
        ("git", ["clean", "-fd"]),
        ("git", ["commit", "-m", "initial"]),
        ("git", ["push", "origin", "main"]),
        ("git", ["pull"]),
        ("git", ["checkout", "main"]),
        ("git", ["switch", "dev"]),
        ("git", ["reset", "--hard"]),
        ("git", ["rebase", "main"]),
        ("git", ["merge", "feature"]),
        ("Start-Process", ["python.exe"]),
    ])
    def test_dangerous_requests(self, program: str, args: list[str]) -> None:
        assert classify_request(_req(program, args)) == RiskLevel.DANGEROUS

    def test_start_process_blocked_target(self) -> None:
        assert classify_request(_req("Start-Process", ["powershell.exe"])) == RiskLevel.BLOCKED

    # ── Blocked programs ─────────────────────────────────────

    @pytest.mark.parametrize("program,args", [
        ("cmd.exe", ["/c", "del", "*"]),
        ("cmd", []),
        ("powershell.exe", []),
        ("Format-Volume", []),
        ("Clear-Disk", []),
    ])
    def test_blocked_requests(self, program: str, args: list[str]) -> None:
        assert classify_request(_req(program, args)) == RiskLevel.BLOCKED

    # ── Meta-character injection blocked ─────────────────────

    @pytest.mark.parametrize("program,args", [
        ("git", ["status;rm", "-rf", "/"]),
        ("python", ["-c", "import os|print(1)"]),
        ("node", ["-e", "console.log`test`"]),
    ])
    def test_metachar_in_args_blocked(self, program: str, args: list[str]) -> None:
        assert classify_request(_req(program, args)) == RiskLevel.BLOCKED
