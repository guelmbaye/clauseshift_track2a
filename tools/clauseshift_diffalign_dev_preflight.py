#!/usr/bin/env python3
"""DEV-only inventory and integrity preflight for SwissGov-RSD DiffAlign references.

Does not score, copy, modify or access full/test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LANGS = ("de", "fr", "it")
SUFFIX = ".jsonl.jsonl"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"{path.name}:{lineno} is not an object")
                yield obj


def catalog(root: Path):
    folder = root / "data/evaluation/encoder_predictions/dev"
    if not folder.is_dir():
        raise ValueError(f"DEV predictions folder unavailable: {folder}")
    by_lang = {}
    for lang in LANGS:
        suffix = f"{lang}{SUFFIX}"
        by_lang[lang] = {
            item.name[:-len(suffix)]: item
            for item in folder.iterdir()
            if item.is_file() and item.name.startswith("DiffAlign(") and item.name.endswith(suffix)
            and item.name[:-len(suffix)].endswith("_admin_")
        }
    common = set.intersection(*(set(by_lang[lang]) for lang in LANGS))
    return by_lang, sorted(common)


def main():
    ap = argparse.ArgumentParser(description="Read-only DiffAlign DEV triplet preflight")
    ap.add_argument("--upstream-root", type=Path, required=True)
    ap.add_argument("--candidate", default="cross_all", help="Substring of model descriptor")
    ap.add_argument("--list", action="store_true", help="List common triplet descriptors")
    args = ap.parse_args()
    root = args.upstream_root.resolve()
    by_lang, common = catalog(root)
    print(f"[DEV ONLY] complete model triplets available={len(common)}")
    if args.list:
        for descriptor in common:
            print("  ", descriptor)
        return
    matches = [p for p in common if args.candidate in p]
    if len(matches) != 1:
        raise SystemExit(f"[FAIL CLOSED] candidate={args.candidate!r} matches {len(matches)} triplets; use --list")
    prefix = matches[0]
    print(f"[CANDIDATE] {prefix}")
    good = True
    for lang in LANGS:
        pred_path = by_lang[lang][prefix]
        gold_path = root / f"data/evaluation/gold_labels/dev/gold_admin_{lang}.jsonl"
        preds, gold = list(read_jsonl(pred_path)), list(read_jsonl(gold_path))
        pred_ids = [r.get("id") for r in preds]
        gold_ids = [r.get("id") for r in gold]
        duplicates = len(set(pred_ids)) != len(pred_ids)
        order_ok = pred_ids == gold_ids
        lengths_ok = 0
        for p, g in zip(preds, gold):
            if (len(p.get("labels_a", [])) == len(g.get("labels_a", []))
                    and len(p.get("labels_b", [])) == len(g.get("labels_b", []))):
                lengths_ok += 1
        status = "PASS" if order_ok and not duplicates and lengths_ok == len(gold) else "REVIEW"
        print(f"[{lang.upper()} {status}] predictions={len(preds)} DEV_gold={len(gold)} "
              f"ids_order={order_ok} duplicate_ids={duplicates} matching_label_lengths={lengths_ok}/{len(gold)}")
        good &= status == "PASS"
    print("[READ ONLY] Consulted encoder_predictions/dev and gold_labels/dev only; no API, no outputs")
    if good:
        print("[READY] To evaluate from SwissGov-RSD root:")
        print(f"python3 -m scripts.evaluate_predictions_admin {str((root / 'data/evaluation/encoder_predictions/dev' / prefix))!r} --split dev")
    else:
        print("[CAUTION] Do not run the official evaluator yet: it may pad or filter mismatched rows.")
        raise SystemExit(2)

if __name__ == "__main__":
    main()
