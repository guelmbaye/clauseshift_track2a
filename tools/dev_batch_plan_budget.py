"""Offline realized batch estimate from existing unlabeled DEV plans only."""
import argparse
import math
from pathlib import Path
from src.core import LANGS,load_jsonl
from src.e2_batch import candidate_batches,_plan_filename
from src.e2_alignment import save_or_load_plan
from types import SimpleNamespace
import json


def main(argv=None):
    p=argparse.ArgumentParser(description='Estimate calls for already materialized DEV plans')
    p.add_argument('--plan-dir',type=Path,default=Path('out/plans/batch_dev_v114'))
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--max-scores-per-call',type=int,default=96)
    p.add_argument('--max-units-per-batch',type=int,default=3)
    args=p.parse_args(argv)
    total=0;docs=0
    for lang in LANGS:
        rows=list(load_jsonl(args.input_dir/f'dev_{lang}.jsonl'))
        planned=[]
        for row in rows:
            path=_plan_filename(args.plan_dir,'dev',lang,row['id'])
            if not path.exists():continue
            # This is a read-only estimator. Plan integrity is checked against its stored hash.
            plan=json.loads(path.read_text(encoding='utf-8'))
            import hashlib
            digest=hashlib.sha256(json.dumps(plan['units'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if digest!=plan.get('plan_sha256'):
                raise ValueError('Corrupt plan checksum '+str(path))
            planned.append(len(candidate_batches(row,plan,{},args.max_scores_per_call,args.max_units_per_batch)))
        print(f'[{lang.upper()} DEV PLANNED] documents={len(planned)}/{len(rows)} '
              f'batch_calls={sum(planned)} min={min(planned) if planned else "—"} '
              f'max={max(planned) if planned else "—"}')
        docs+=len(planned);total+=sum(planned)
    print(f'[OFFLINE ONLY] planned documents={docs} batch_calls={total}; excludes retries/provider rate-limits')

if __name__=='__main__':main()
