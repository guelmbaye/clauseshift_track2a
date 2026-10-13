"""DEV-only exploratory Spearman: E0-PJQ vs completed E1-I documents.
This is NOT the official UZH leaderboard/evaluation result.
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from src.core import load_jsonl
from src.e1i import completed_document, read_updates, signature
from tools.compare_e0_e1c_dev import spearman


def main(argv=None):
    p=argparse.ArgumentParser(description='Compare E0-PJQ with E1-I on common DEV documents')
    p.add_argument('--lang',choices=('de','fr','it'),required=True)
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--e0-checkpoints',type=Path,default=Path('out/checkpoints/e0pjq_publicai_4096'))
    p.add_argument('--e1i-checkpoints',type=Path,default=Path('out/checkpoints/e1i_dev_v1'))
    p.add_argument('--chunk-tokens',type=int,default=24)
    p.add_argument('--context',type=int,default=20)
    p.add_argument('--max-tokens',type=int,default=512)
    args=p.parse_args(argv)
    try:
        gold_path=args.upstream_root/'data'/'evaluation'/'gold_labels'/'dev'/f'gold_admin_{args.lang}.jsonl'
        if not gold_path.is_file():
            raise ValueError(f'Official DEV gold missing: {gold_path}')
        gold={r['id']:r for r in load_jsonl(gold_path)}
        inp={r['id']:r for r in load_jsonl(args.input_dir/f'dev_{args.lang}.jsonl')}
        e0_path=args.e0_checkpoints/f'dev_{args.lang}.checkpoint.jsonl'
        e0={r['id']:r for r in load_jsonl(e0_path)} if e0_path.exists() else {}
        model=os.environ.get('LLM_NAME','');url=os.environ.get('LLM_BASE_URL','')
        if not model or not url:raise ValueError('Set LLM_NAME and LLM_BASE_URL for checkpoint signature verification')
        settings=SimpleNamespace(chunk_tokens=args.chunk_tokens,context=args.context,max_tokens=args.max_tokens)
        sig=signature(settings,model,url)
        state=read_updates(args.e1i_checkpoints/f'dev_{args.lang}_indexed.checkpoint.jsonl',sig,inp,settings)
        shared=[]
        for rowid,row in inp.items():
            if rowid not in e0 or rowid not in gold:continue
            pred=completed_document(row,state,settings)
            if pred is not None:
                shared.append((rowid,gold[rowid],e0[rowid],pred))
        if not shared:
            print('[DEV ONLY] No completed common E0/E1-I documents yet')
            return 0
        xs=[];ys0=[];ys1=[]
        for rowid,g,e0pred,e1pred in shared:
            for side in ('labels_a','labels_b'):
                if any(len(r[side])!=len(g[side]) for r in (e0pred,e1pred)):
                    raise ValueError(f'Length mismatch {rowid}/{side}')
                for label,v0,v1 in zip(g[side],e0pred[side],e1pred[side]):
                    if label==-1:continue
                    if v0==-1 or v1==-1:raise ValueError('Negative score on semantic token')
                    xs.append(label);ys0.append(v0);ys1.append(v1)
        print(f'[DEV EXPLORATORY] {args.lang} docs={len(shared)} tokens={len(xs)}')
        print('E0-PJQ Spearman proxy:',spearman(xs,ys0))
        print('E1-I Spearman proxy:',spearman(xs,ys1))
        print('[CAUTION] Small overlap is biased/unstable; NOT official UZH DEV or test score.')
        print('IDs:',', '.join(rowid for rowid,*_ in shared[:12]))
        return 0
    except (ValueError,KeyError,FileNotFoundError) as exc:
        print(f'[FAIL CLOSED] {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
