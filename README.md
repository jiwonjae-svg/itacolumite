# Itacolumite

```text
+------------------------------------------------------------------+
|  III TTTT   A    CCCC  OOO   L    U   U M   M III TTTT EEEEE     |
|   I    T   A A  C     O   O  L    U   U MM MM  I    T   E         |
|   I    T  AAAAA C     O   O  L    U   U M M M  I    T   EEEE      |
|   I    T  A   A C     O   O  L    U   U M   M  I    T   E         |
|  III   T  A   A  CCCC  OOO   LLLL  UUU  M   M III   T   EEEEE     |
|                                                                  |
|        Autonomous computer-use agent on native Windows           |
+------------------------------------------------------------------+
```

> **Itacolumite**(이타콜루마이트) — 유연한 사암처럼 단단하면서도 유연하게 적응하는 자율 컴퓨터 제어 AI 에이전트

## Overview

Itacolumite는 **네이티브 Windows 컴퓨터**를 자율적으로 제어하는 AI 에이전트입니다.
Windows API를 통해 화면을 직접 인식하고 마우스·키보드를 조작하여 소프트웨어 개발의 전 과정을 자동으로 수행합니다.

현재 계획 기준:
- 단일 모니터를 전제로 동작합니다.
- 명령 실행은 PowerShell만 사용합니다.
- 상태 출력은 한 터미널에서 보고, 제어 명령은 별도 PowerShell에서 보냅니다.

## Quick Start

```powershell
# 1. Clone & setup
git clone https://github.com/jiwonjae-svg/itacolumite
cd itacolumite

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Copy environment file
Copy-Item .env.example .env
# Edit .env and set your GEMINI_API_KEY

# 5. Run the agent in PowerShell #1
itacolumite task "Create a Python Flask TODO API with tests"

# 6. Send control commands from PowerShell #2
itacolumite control pause
itacolumite control send "VS Code부터 먼저 열어"
itacolumite control resume
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

Detailed planning document: [`docs/plan.md`](docs/plan.md)
