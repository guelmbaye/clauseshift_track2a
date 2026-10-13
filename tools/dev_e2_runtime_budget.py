"""Offline DEV-only approximate API-call budget, never examines gold/final/test."""
import argparse
import math
from pathlib import Path
from statistics import median

from src.core import LANGS, load_jsonl, validate_input_rows, is_punctuation_only


def counts(row):
    return sum(not is_punctuation_only(tok) for side in ('text_a','text_b')
               for tok in row[side].split())


def estimate(input_dir, cap, seconds, concurrency):
    if cap < 2 or seconds <= 0 or concurrency < 1:
        raise ValueError('Use cap >=2, seconds > 0 and concurrency >= 1')
    total_docs=0
    bound_calls=0
    for lang in LANGS:
        path=Path(input_dir)/f'dev_{lang}.jsonl'
        docs=list(load_jsonl(path))
        validate_input_rows(docs)
        values=[counts(row) for row in docs]
        # Optimistic lower bound: ignores segmentation overhead, retries, and rate limits.
        requests=sum(math.ceil(n/cap) for n in values)
        total_docs+=len(values);bound_calls+=requests
        print(f'[{lang.upper()} DEV] documents={len(values)} nonpunct_p50={median(values):.1f} '
              f'optimistic_min_calls={requests} '
              f'min_minutes_sequential={requests*seconds/60:.1f}')
    print(f'[DEV LOWER BOUND] documents={total_docs} calls>={bound_calls} '
          f'wall_hours>={bound_calls*seconds/3600/concurrency:.2f} '
          f'at {seconds:g}s/call and {concurrency} independent concurrent worker(s)')
    print('[CAUTION] Counts exclude plan overhead and failed/retried calls. Not a runtime guarantee. '
          'Extrapolation to full 224 documents/language is unsupported by this DEV-only estimate.')
    print('[READ ONLY] Only DEV unlabeled inputs accessed, no test/final/gold/API/checkpoint.')


def main(argv=None):
    p=argparse.ArgumentParser(description='Estimate optimistic indexed API budget from DEV unlabeled inputs')
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--max-scores-per-call',type=int,default=36)
    p.add_argument('--seconds-per-call',type=float,default=15)
    p.add_argument('--concurrency',type=int,default=1)
    a=p.parse_args(argv)
    estimate(a.input_dir,a.max_scores_per_call,a.seconds_per_call,a.concurrency)

if __name__=='__main__':
    main()
