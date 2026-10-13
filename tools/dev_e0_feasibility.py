#!/usr/bin/env python3
"""Offline DEV-only feasibility report. NEVER loads full/test/gold data or makes HTTP calls."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

LANGS=('de','fr','it')
BUCKETS=((0,128),(129,256),(257,512),(513,768),(769,1024),(1025,10**12))

def report(input_dir=Path('data/inputs'), checkpoint_dir=Path('out/checkpoints/e0pjq_publicai_4096')):
    result={}
    for lang in LANGS:
        path=Path(input_dir)/f'dev_{lang}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing DEV-only input file: {path}. Never substitute full/test data.')
        rows=[]
        seen=set()
        with path.open(encoding='utf-8') as h:
            for n,line in enumerate(h,1):
                if not line.strip():continue
                item=json.loads(line)
                if item['id'] in seen:raise ValueError(f'Duplicate ID in {path}: {item["id"]}')
                seen.add(item['id'])
                rows.append({'id':item['id'],'tokens_a':len(item['text_a'].split()),
                             'tokens_b':len(item['text_b'].split())})
        if not rows: raise ValueError(f'Empty DEV file: {path}')
        completed=set()
        cp=Path(checkpoint_dir)/f'dev_{lang}.checkpoint.jsonl'
        if cp.exists():
            with cp.open(encoding='utf-8') as h:
                for line in h:
                    if line.strip():
                        item=json.loads(line)
                        if item['id'] in completed:raise ValueError(f'Duplicate checkpoint: {item["id"]}')
                        completed.add(item['id'])
            if not completed.issubset(seen):raise ValueError(f'Unknown DEV checkpoint ID for {lang}')
        for r in rows:r['total_tokens']=r['tokens_a']+r['tokens_b']
        lengths=sorted(r['total_tokens'] for r in rows)
        pending=[r for r in rows if r['id'] not in completed]
        hist=[]
        for lo,hi in BUCKETS:
            choices=[r for r in pending if lo<=r['total_tokens']<=hi]
            representative=min(choices,key=lambda r:(abs(r['total_tokens']-(lo+hi)/2),r['total_tokens'],r['id'])) if choices else None
            # For open-ended bucket, pick smallest available length, not absurd midpoint.
            if hi>10**9 and choices:
                representative=min(choices,key=lambda r:(r['total_tokens'],r['id']))
            hist.append({'bucket':f'{lo}-{hi}' if hi<10**9 else f'{lo}+',
                         'pending':len(choices), 'probe_id':representative['id'] if representative else None,
                         'probe_tokens':representative['total_tokens'] if representative else None})
        def quantile(p):return lengths[min(len(lengths)-1,math.ceil(p*len(lengths))-1)]
        result[lang]={'all':len(rows),'completed':len(completed),'pending':len(pending),
                      'p50':quantile(.50),'p90':quantile(.90),'max':lengths[-1],
                      'buckets':hist}
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description='Inspect DEV-only long-document feasibility; zero API calls')
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--checkpoint-dir',type=Path,default=Path('out/checkpoints/e0pjq_publicai_4096'))
    a=p.parse_args(argv)
    results=report(a.input_dir,a.checkpoint_dir)
    for lang,stats in results.items():
        print(f'[{lang.upper()} DEV] complete={stats["completed"]}/{stats["all"]} pending={stats["pending"]} '
              f'p50={stats["p50"]} p90={stats["p90"]} max={stats["max"]} whitespace tokens')
        for b in stats['buckets']:
            print(f'  tokens={b["bucket"]:>8} pending={b["pending"]:>3} '
                  f'probe={b["probe_id"] or "—"} n={b["probe_tokens"] if b["probe_tokens"] is not None else "—"}')
    print('[DEV ONLY] No test/final/gold files accessed; no API calls or checkpoint changes.')

if __name__=='__main__':main()
