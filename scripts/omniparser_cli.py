from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin CLI wrapper for Microsoft OmniParser")
    parser.add_argument("--image", required=True, help="Path to the input screenshot")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--repo-root", help="Path to the cloned OmniParser repository")
    parser.add_argument("--som-model-path", default="weights/icon_detect/model.pt")
    parser.add_argument("--caption-model-name", default="florence2")
    parser.add_argument("--caption-model-path", default="weights/icon_caption_florence")
    parser.add_argument("--box-threshold", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--log-path", help="Optional persistent progress log path")
    return parser.parse_args()


def _resolve_repo_root(raw_repo_root: str | None) -> Path:
    if raw_repo_root:
        return Path(raw_repo_root).resolve()
    return Path(os.getcwd()).resolve()


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve_log_path(args: argparse.Namespace) -> Path:
    if args.log_path:
        return Path(args.log_path).resolve()
    image_path = Path(args.image).resolve()
    return Path(str(image_path) + ".omniparser.log")


def _log(message: str, log_path: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, file=sys.stderr, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    log_path = _resolve_log_path(args)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ["OMNIPARSER_PROGRESS_LOG_PATH"] = str(log_path)
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    repo_root = _resolve_repo_root(args.repo_root)
    sys.path.insert(0, str(repo_root))

    _log(f"wrapper start repo_root={repo_root} image={Path(args.image).resolve()} device={args.device} batch_size={args.batch_size} imgsz={args.imgsz}", log_path)

    import_started = time.perf_counter()
    from util.omniparser import Omniparser
    _log(f"imported util.omniparser in {time.perf_counter() - import_started:.2f}s", log_path)

    image_started = time.perf_counter()
    image_bytes = Path(args.image).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    _log(f"read image bytes in {time.perf_counter() - image_started:.2f}s size_bytes={len(image_bytes)}", log_path)

    init_started = time.perf_counter()
    parser = Omniparser(
        {
            "som_model_path": args.som_model_path,
            "caption_model_name": args.caption_model_name,
            "caption_model_path": args.caption_model_path,
            "BOX_TRESHOLD": args.box_threshold,
            "device": args.device,
            "batch_size": max(args.batch_size, 1),
            "imgsz": max(args.imgsz, 64),
        }
    )
    _log(f"constructed Omniparser in {time.perf_counter() - init_started:.2f}s", log_path)

    parse_started = time.perf_counter()
    _som_image_base64, parsed_content_list = parser.parse(image_base64)
    _log(f"parse completed in {time.perf_counter() - parse_started:.2f}s elements={len(parsed_content_list)}", log_path)
    payload = {"parsed_content_list": parsed_content_list}
    serialized = json.dumps(payload, ensure_ascii=False, default=_json_default)

    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
        _log(f"wrote output json to {Path(args.output).resolve()}", log_path)
        return
    print(serialized)


if __name__ == "__main__":
    main()