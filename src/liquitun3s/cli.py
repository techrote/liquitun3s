from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import validate_vl_manifest
from .hardware import print_preflight
from .registry import aliases, model_entry
from .source import resolve_revision


def _cmd_models(_args: argparse.Namespace) -> int:
    for alias in aliases():
        entry = model_entry(alias)
        print(
            f"{alias:10} {entry['id']:28} {entry['kind']:10} "
            f"{entry['local_training']:14} {entry['status']}"
        )
    return 0


def _cmd_source(args: argparse.Namespace) -> int:
    print(json.dumps(resolve_revision(args.model, args.revision), indent=2))
    return 0


def _cmd_validate_vl(args: argparse.Namespace) -> int:
    result = validate_vl_manifest(
        Path(args.dataset),
        allow_missing_images=args.allow_missing_images,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_train_vl(args: argparse.Namespace) -> int:
    from .train_vl import train_vl

    train_vl(Path(args.config))
    return 0


def _cmd_train_text(args: argparse.Namespace) -> int:
    from .train_text import train_text

    train_text(Path(args.config))
    return 0


def _cmd_vl(args: argparse.Namespace) -> int:
    from .eval_vl import run_vl

    result = run_vl(
        image_path=Path(args.image),
        prompt=args.prompt,
        adapter=Path(args.adapter) if args.adapter else None,
        revision=args.revision,
        max_image_side=args.max_image_side,
        max_new_tokens=args.max_new_tokens,
        thinking=args.thinking == "on",
        record_path=Path(args.record) if args.record else None,
    )
    print(result["output"])
    print(json.dumps({key: value for key, value in result.items() if key != "output"}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liquitun3s",
        description="Local Titan RTX training/evaluation harness for Liquid AI LFM3.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="List registered model lanes.")
    models.set_defaults(func=_cmd_models)

    source = sub.add_parser("source", help="Resolve and verify an exact Hugging Face source.")
    source.add_argument("model", choices=aliases())
    source.add_argument("--revision")
    source.set_defaults(func=_cmd_source)

    preflight = sub.add_parser("preflight", help="Check local CUDA/Titan training readiness.")
    preflight.set_defaults(func=lambda _args: print_preflight())

    validate = sub.add_parser("validate-vl", help="Validate a Cybersand VLM JSONL manifest.")
    validate.add_argument("dataset")
    validate.add_argument("--allow-missing-images", action="store_true")
    validate.set_defaults(func=_cmd_validate_vl)

    train_vl_parser = sub.add_parser("train-vl", help="Run local FP16 LoRA SFT for LFM3-VL.")
    train_vl_parser.add_argument("config")
    train_vl_parser.set_defaults(func=_cmd_train_vl)

    train_text_parser = sub.add_parser("train-text", help="Run local FP16 LoRA SFT for LFM3 text.")
    train_text_parser.add_argument("config")
    train_text_parser.set_defaults(func=_cmd_train_text)

    vl = sub.add_parser("vl", help="Run one local LFM3-VL image prompt.")
    vl.add_argument("image")
    vl.add_argument("--prompt", required=True)
    vl.add_argument("--adapter")
    vl.add_argument("--revision")
    vl.add_argument("--thinking", choices=["on", "off"], default="off")
    vl.add_argument("--max-image-side", type=int, default=768)
    vl.add_argument("--max-new-tokens", type=int, default=256)
    vl.add_argument("--record")
    vl.set_defaults(func=_cmd_vl)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
