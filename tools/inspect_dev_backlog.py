#!/usr/bin/env python3
"""Zero-cost DEV-only input and checkpoint inspection. Never touches test or final."""
import argparse
import json
from pathlib import Path
def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f'{path}:{n}: invalid JSON: {exc}') from exc


def describe(row):
    a=len(row['text_a'].split());b=len(row['text_b'].split())
    return f"{row['id']}: EN={a}, target={b}, total={a+b} whitespace-tokens, chars={len(row['text_a'])+len(row['text_b'])}"


def main():
    p=argparse.ArgumentParser(description='Inspect pending DEV rows without API calls')
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--checkpoint-dir',type=Path,default=Path('out/checkpoints/e0p_publicai_4096'))
    p.add_argument('--top',type=int,default=5)
    args=p.parse_args()
    if args.top<1 or args.top>100:raise SystemExit('--top must be between 1 and 100')
    for lang in ('de','fr','it'):
        inp=args.input_dir/f'dev_{lang}.jsonl'
        if not inp.is_file():raise SystemExit(f'Missing DEV inputs: {inp}')
        rows=list(load_jsonl(inp))
        cp=args.checkpoint_dir/f'dev_{lang}.checkpoint.jsonl'
        complete=set()
        if cp.exists():
            for r in load_jsonl(cp):
                if r['id'] in complete:raise SystemExit(f'duplicate checkpoint ID {r["id"]}')
                complete.add(r['id'])
        if not complete.issubset({r['id'] for r in rows}):raise SystemExit('unknown ID in checkpoint')
        todo=[r for r in rows if r['id'] not in complete]
        print(f'\n[{lang.upper()}] completed={len(complete)}/{len(rows)} pending={len(todo)}')
        print('FIRST PENDING IN REFERENCE ORDER:')
        for r in todo[:args.top]:print('  '+describe(r))
        print('SHORTEST PENDING (DEV ONLY):')
        for r in sorted(todo,key=lambda r:(len(r['text_a'].split())+len(r['text_b'].split()),r['id']))[:args.top]:
            print('  '+describe(r))
    print('\n[SAFE] DEV data/checkpoints inspected; no API calls; no test access or output changes.')

if __name__=='__main__':main()
