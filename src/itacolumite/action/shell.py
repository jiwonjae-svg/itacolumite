"""Shell execution via PowerShell subprocess with strict policy enforcement.

Rules:
- shell_exec accepts structured {program, args} only.
- PowerShell meta-characters (;, |, >, >>) and Invoke-Expression are blocked.
- cmd.exe is never used.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from enum import Enum

from itacolumite.config.settings import get_settings

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    SAFE = "safe"
    NORMAL = "normal"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


# ── 메타 문자 차단 ───────────────────────────────────────────

_META_CHARS = re.compile(r"[;|><`$]")
_BLOCKED_CMDLETS = re.compile(
    r"\b(Invoke-Expression|iex|Invoke-Command|icm)\b", re.IGNORECASE
)


def _has_metachar(token: str) -> bool:
    return bool(_META_CHARS.search(token))


def _normalize_token(token: str) -> str:
    return token.strip().casefold()


# ── 정책 목록 ────────────────────────────────────────────────

_PROGRAM_ALIASES: dict[str, str] = {
    "ls": "get-childitem",
    "dir": "get-childitem",
    "cat": "get-content",
    "type": "get-content",
    "pwd": "get-location",
    "md": "mkdir",
}


def _normalize_program(program: str) -> str:
    normalized = _normalize_token(program)
    return _PROGRAM_ALIASES.get(normalized, normalized)


_SAFE_PROGRAMS: set[str] = {
    "get-childitem", "get-content", "get-location", "set-location",
    "test-path", "resolve-path", "get-item", "get-itemproperty",
    "select-string", "measure-object", "get-process",
    "mkdir",
    "git", "python", "node",
    "where.exe", "whoami",
}

_SAFE_GIT_SUBCOMMANDS: set[str] = {
    "status", "log", "diff", "show", "branch", "remote", "rev-parse", "describe", "tag",
}

_NORMAL_PROGRAMS: set[str] = {
    "pytest", "pip", "npm", "cargo", "go", "dotnet", "make", "cmake",
    "java", "javac", "gradle", "mvn", "ruff", "mypy",
    "code", "code.cmd", "code-insiders", "code-insiders.cmd",
    "notepad", "notepad.exe", "calc", "calc.exe", "explorer", "explorer.exe",
}

_NORMAL_LAUNCH_TARGETS: set[str] = {
    "code", "code.cmd", "code-insiders", "code-insiders.cmd",
    "notepad", "notepad.exe", "calc", "calc.exe", "explorer", "explorer.exe",
}

_BLOCKED_PROGRAMS: set[str] = {
    "clear-disk", "format-volume", "diskpart", "bcdedit", "sfc", "dism",
    "stop-computer", "restart-computer", "shutdown", "logoff",
    "reg", "regedit",
    "invoke-expression", "iex",
    # 쉘 인터프리터 직접 호출 차단 (구조화된 명령 우회 방지)
    "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    "wscript", "cscript", "mshta",
}

_DANGEROUS_PROGRAMS: set[str] = {
    "remove-item", "stop-process", "npm",  # npm publish 등
}


@dataclass
class ShellRequest:
    """구조화된 셸 실행 요청."""
    program: str
    args: list[str]
    cwd: str | None = None
    timeout: int = 30


@dataclass
class ShellResult:
    """셸 실행 결과."""
    success: bool
    exit_code: int
    output: str
    error: str = ""


def classify_request(request: ShellRequest) -> RiskLevel:
    """프로그램 + args 기반 리스크 분류."""
    prog = _normalize_program(request.program)

    # 메타 문자 검사
    if _has_metachar(request.program) or any(_has_metachar(a) for a in request.args):
        return RiskLevel.BLOCKED

    # Invoke-Expression 패턴
    if _BLOCKED_CMDLETS.search(prog) or any(_BLOCKED_CMDLETS.search(a) for a in request.args):
        return RiskLevel.BLOCKED

    if prog == "start-process":
        if not request.args:
            return RiskLevel.DANGEROUS
        target = _normalize_program(request.args[0])
        if target in _BLOCKED_PROGRAMS:
            return RiskLevel.BLOCKED
        if target in _NORMAL_LAUNCH_TARGETS:
            return RiskLevel.NORMAL
        return RiskLevel.DANGEROUS

    if prog in _BLOCKED_PROGRAMS:
        return RiskLevel.BLOCKED

    if prog in _SAFE_PROGRAMS:
        # git의 경우 서브커맨드 확인
        if prog == "git" and request.args:
            subcommand = _normalize_token(request.args[0])
            if subcommand in _SAFE_GIT_SUBCOMMANDS:
                return RiskLevel.SAFE
            if subcommand in ("add", "fetch"):
                return RiskLevel.NORMAL
            if subcommand in ("commit", "push", "pull", "checkout", "switch", "reset", "rebase", "merge"):
                return RiskLevel.DANGEROUS
            return RiskLevel.DANGEROUS
        return RiskLevel.SAFE

    if prog in _NORMAL_PROGRAMS:
        return RiskLevel.NORMAL

    if prog in _DANGEROUS_PROGRAMS:
        return RiskLevel.DANGEROUS

    # 알 수 없는 프로그램은 DANGEROUS
    return RiskLevel.DANGEROUS


def _build_ps_command(request: ShellRequest) -> str:
    """ShellRequest → PowerShell -Command 문자열."""
    parts = [request.program] + request.args
    # 각 토큰을 single-quote로 감싸서 인젝션 방지
    escaped = []
    for part in parts:
        # PowerShell single-quote 이스케이프: ' → ''
        safe = part.replace("'", "''")
        escaped.append(f"'{safe}'")
    return "& " + " ".join(escaped)


class ShellController:
    """PowerShell 기반 구조화된 명령 실행."""

    def __init__(self) -> None:
        self._settings = get_settings()

    def execute(self, request: ShellRequest, *, force: bool = False) -> ShellResult:
        """정책 검사 후 PowerShell 실행."""
        risk = classify_request(request)

        if risk == RiskLevel.BLOCKED:
            reason = f"BLOCKED: {request.program} {request.args}"
            logger.warning(reason)
            return ShellResult(success=False, exit_code=-1, output="", error=reason)

        if risk == RiskLevel.DANGEROUS and not force:
            reason = f"DANGEROUS (requires approval): {request.program} {request.args}"
            logger.warning(reason)
            return ShellResult(success=False, exit_code=-1, output="", error=reason)

        ps_command = _build_ps_command(request)
        logger.info("[%s] Executing: %s", risk.value, ps_command)

        try:
            result = subprocess.run(
                [
                    self._settings.agent.shell_executable,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy", "Bypass",
                    "-Command", ps_command,
                ],
                shell=False,
                capture_output=True,
                text=True,
                timeout=request.timeout,
                cwd=request.cwd,
            )
            return ShellResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                output=result.stdout.strip(),
                error=result.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return ShellResult(
                success=False, exit_code=-1, output="", error=f"Timeout ({request.timeout}s)"
            )
        except Exception as e:
            return ShellResult(success=False, exit_code=-1, output="", error=str(e))
