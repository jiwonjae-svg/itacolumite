param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.14"
)

$ErrorActionPreference = "Stop"

function Test-ItacolumiteMixedVenv {
    if (-not (Test-Path ".\\.venv")) {
        return $false
    }

    if (Test-Path ".\\.venv\\bin") {
        return $true
    }

    if (-not (Test-Path ".\\.venv\\Scripts\\python.exe")) {
        return $true
    }

    if (Test-Path ".\\.venv\\pyvenv.cfg") {
        $cfg = Get-Content ".\\.venv\\pyvenv.cfg" -Raw
        if ($cfg -match "msys64|mingw64|\\bin\\python.exe") {
            return $true
        }
    }

    return $false
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    $pythonCommand = @("-$PythonVersion")

    if (Test-ItacolumiteMixedVenv) {
        Remove-Item .\.venv -Recurse -Force
    }

    if (-not (Test-Path ".\\.venv\\Scripts\\python.exe")) {
        & $PythonLauncher @pythonCommand -m venv .venv
    }

    if (Test-ItacolumiteMixedVenv) {
        throw "The .venv environment was not created as a Windows CPython virtual environment. Use the Python launcher managed by 'py' and avoid MSYS/MinGW Python for this repository."
    }

    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
}
finally {
    Pop-Location
}