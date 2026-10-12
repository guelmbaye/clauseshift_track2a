#!/usr/bin/env python3
"""DEV-only, read existing saved raw answer; make no network requests or checkpoints."""
import argparse, json
from pathlib import Path
from src.core import load_jsonl, normalize_prediction, PredictionError

def main():
    p=argparse.ArgumentParser(description='Offline punctuation repair proof using saved DEV raw response')
    p.add_argument('--raw-file',type=Path,required=True)
    p.add_argument('--lang',choices=('de','fr','it'),required=True)
    p.add_argument('--id',required=True)
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    a=p.parse_args()
    rows=list(load_jsonl(a.input_dir/f'dev_{a.lang}.jsonl'))
    row=next((r for r in rows if r['id']==a.id),None)
    if row is None:
        raise SystemExit('DEV document ID not found in dev inputs')
    raw=a.raw_file.read_text(encoding='utf-8')
    report=[]
    try:
        result=normalize_prediction(row,raw,punctuation_repair=True,repair_log=report)
    except PredictionError as e:
        print(f'[REFUSED] {e}')
        return 2
    print(f'[REPAIRABLE] {a.lang}/{a.id}, no semantic token was changed or guessed')
    for item in report:
        print(f"  {item['side']}: changed={item['changed']}; source_punctuation={item['source_punctuation']}; generated_punctuation={item['generated_punctuation']}; semantic_tokens_preserved={item['semantic_tokens_preserved']}")
    print(f"[VALID] labels_a={len(result['labels_a'])}; labels_b={len(result['labels_b'])}")
    print('[SAFE] Offline DEV check only; no prediction/checkpoint file written; no API requests.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
