# Itacolumite

## Overview

Itacolumite는 **네이티브 Windows 컴퓨터**를 자율적으로 제어하는 AI 에이전트입니다.
Windows API를 통해 화면을 직접 인식하고 마우스·키보드를 조작하여 소프트웨어 개발의 전 과정을 자동으로 수행합니다.

현재 계획 기준:
- 기본은 주 모니터 기준으로 동작하고, `AGENT_CAPTURE_TARGET=virtual-desktop`로 다중 모니터 전체 가상 데스크톱을 캡처할 수 있습니다.
- 명령 실행은 PowerShell만 사용합니다.
- 상태 출력은 한 터미널에서 보고, 제어 명령은 별도 PowerShell에서 보냅니다.

## Quick Start

```powershell
# 1. Clone & setup
git clone https://github.com/jiwonjae-svg/itacolumite
cd itacolumite

# 2. Create virtual environment with the Windows Python launcher
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip first so Python 3.14 can resolve pywin32 wheels
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -e ".[dev]"

# Or run the one-shot bootstrap helper
.\scripts\bootstrap.ps1

# 5. Copy environment file
Copy-Item .env.example .env
# Edit .env and set your GEMINI_API_KEY

# 6. Run the agent in PowerShell #1
itacolumite task "Create a Python Flask TODO API with tests"

# 7. Send control commands from PowerShell #2
itacolumite control pause
itacolumite control send "VS Code부터 먼저 열어"
itacolumite control resume

# 8. Optional grounding utilities
itacolumite grounding extract-text
itacolumite grounding run-omniparser
itacolumite grounding report
```

## Architecture

```
Windows Host (직접 실행)
  ├── Itacolumite Agent (Python)
  │     ├── Gemini 2.5 Flash API (Vision + Reasoning)
  │     ├── Windows API (Screen capture + Input control)
  │     ├── PowerShell execution
  │     └── CLI Status Output + Named Pipe Control
  │
  ├── PowerShell Control Client
  │
  └── Native Applications
        ├── VS Code + GitHub Copilot
        ├── Google Chrome
        ├── PowerShell / Windows Terminal
        └── Development Tools
```

## Documentation

Planning notes and implementation brainstorms are maintained as local internal documents under `internal/docs/` and are not part of the public-facing repository documentation.

## Grounding Utilities

Use the built-in grounding helpers when you want extra OCR-style evidence or a quick threshold-tuning report.

```powershell
# Capture the current screen and generate a provider JSON file under agent-data/grounding/providers
itacolumite grounding extract-text

# Capture the current screen and run a local OmniParser command into the same provider format
itacolumite grounding run-omniparser

# Summarize validation_events.jsonl and write an HTML report under agent-data/grounding/reports
itacolumite grounding report
```

By default, pointer actions refresh a Gemini OCR grounding provider before validation so the validator can snap broad model bboxes onto tighter OCR anchors.

If you want to turn that behavior off to reduce extra vision calls, set these in `.env`:

```dotenv
GROUNDING_ENABLE_GEMINI_OCR_PROVIDER=false
GROUNDING_AUTO_REFRESH_GEMINI_OCR=false
```

To enable a local OmniParser executable, configure the command and argument template in `.env`. `{image_path}` and `{output_path}` are replaced automatically.

```dotenv
GROUNDING_ENABLE_OMNIPARSER_RUNNER=true
GROUNDING_AUTO_REFRESH_OMNIPARSER=true
GROUNDING_OMNIPARSER_COMMAND=python
GROUNDING_OMNIPARSER_ARGS=C:\\tools\\omniparser_cli.py --image "{image_path}" --output "{output_path}" --device cpu --batch-size 8 --imgsz 640 --log-path "{image_path}.omniparser.log"
GROUNDING_OMNIPARSER_OUTPUT_NAME=omniparser_latest.json
```

Start with `--device cpu --batch-size 8 --imgsz 640` for the first smoke test. After that is stable, move to `--device cuda` or a larger batch size gradually.
The progress log is written next to the captured screenshot as `*.omniparser.log`, so if a run stalls you can inspect the last completed phase without attaching a debugger.

If you use multiple monitors and want the agent to reason over the entire virtual desktop instead of only the primary display, set:

```dotenv
AGENT_CAPTURE_TARGET=virtual-desktop
```

## Troubleshooting

If `pip install -e ".[dev]"` fails with `No matching distribution found for pywin32>=306`, there are two common causes on this repository:

1. `pip` is too old to resolve current Python 3.14 wheels.
2. `.venv` was created from MSYS2/MinGW Python instead of Windows CPython.

```powershell
py -0p
Remove-Item .\.venv -Recurse -Force
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If `.venv\pyvenv.cfg` points at `C:\msys64\mingw64\bin\python.exe` or your environment contains `.venv\bin`, recreate the virtual environment with `py -3.14 -m venv .venv`. This project expects a native Windows CPython environment.
