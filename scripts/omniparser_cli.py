from __future__ import annotations

import argparse
import base64
import json
import os
import sys
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


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    repo_root = _resolve_repo_root(args.repo_root)
    sys.path.insert(0, str(repo_root))

    from util.omniparser import Omniparser

    image_bytes = Path(args.image).read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

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
    _som_image_base64, parsed_content_list = parser.parse(image_base64)
    payload = {"parsed_content_list": parsed_content_list}
    serialized = json.dumps(payload, ensure_ascii=False, default=_json_default)

    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
        return
    print(serialized)


if __name__ == "__main__":
    main()