"""Inspect a frozen E2 DEV alignment plan without gold labels or API calls."""
import argparse
import sys
from pathlib import Path
from src.core import load_jsonl
from src.e2_alignment import validate_units, alignment_summary, plan_path
import json


def main(argv=None):
    p=argparse.ArgumentParser(description='DEV-only offline visual inspection of E2 matched and unaligned windows')
    p.add_argument('--lang',choices=('de','fr','it'),required=True)
    p.add_argument('--doc-id',required=True)
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--plan-dir',type=Path,required=True)
    p.add_argument('--limit',type=int,default=15)
    args=p.parse_args(argv)
    try:
        doc=next(r for r in load_jsonl(args.input_dir/f'dev_{args.lang}.jsonl') if r['id']==args.doc_id)
        a,b=doc['text_a'].split(),doc['text_b'].split()
        plan=json.loads(plan_path(args.plan_dir,args.lang,args.doc_id).read_text(encoding='utf-8'))
        validate_units(plan['units'],len(a),len(b),plan['settings']['chunk_tokens'])
        result=alignment_summary(plan,doc)
        print(f'[E2 DEV PLAN] {args.doc_id} lang={args.lang} '+json.dumps(result))
        print(f'plan_sha256={plan["plan_sha256"]}')
        for i,u in enumerate(plan['units'][:args.limit],1):
            ra,rb=u['ra'],u['rb']
            print(f'\nUNIT {i}/{len(plan["units"])} {u["kind"]} similarity={u["similarity"]}')
            print(f'EN {ra}: ', ' '.join(a[ra[0]:ra[1]]))
            print(f'{args.lang.upper()} {rb}: ', ' '.join(b[rb[0]:rb[1]]))
        print('\n[READ ONLY] No gold, predictions or checkpoint contents accessed')
        return 0
    except (ValueError,KeyError,FileNotFoundError,StopIteration,TypeError) as exc:
        print(f'[FAIL CLOSED] {exc}',file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
