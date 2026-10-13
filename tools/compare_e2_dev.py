"""Exploratory E0-PJQ / E2 DEV-only comparison on COMPLETE overlapping documents."""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from src.core import load_jsonl
from src.e2 import fingerprint, read_document_updates, completed_document
from src.e2_alignment import save_or_load_plan, plan_path
from tools.compare_e0_e1c_dev import spearman


def main(argv=None):
    p=argparse.ArgumentParser(description='DEV-only E0 vs E2 exploratory quality diagnostics')
    p.add_argument('--lang',choices=('de','fr','it'),required=True)
    p.add_argument('--doc-id',default='')
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--e0-checkpoints',type=Path,default=Path('out/checkpoints/e0pjq_publicai_4096'))
    p.add_argument('--e2-checkpoints',type=Path,default=Path('out/checkpoints/e2a_dev_v1'))
    p.add_argument('--plan-dir',type=Path,default=Path('out/plans/e2a_dev_v1'))
    p.add_argument('--alignment',choices=['semantic','semantic_coverage','proportional'],default='semantic')
    p.add_argument('--prompt',choices=['short','fewshot'],default='short')
    p.add_argument('--embedding-model',default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    p.add_argument('--embedding-revision',default='main')
    p.add_argument('--max-unit-tokens',type=int,default=20)
    p.add_argument('--similarity-threshold',type=float,default=0.46)
    p.add_argument('--merge-penalty',type=float,default=0.06)
    p.add_argument('--coverage-weight',type=float,default=0.50)
    p.add_argument('--anchor-bonus',type=float,default=0.35)
    p.add_argument('--chunk-tokens',type=int,default=48)
    p.add_argument('--context',type=int,default=20)
    p.add_argument('--max-tokens',type=int,default=512)
    args=p.parse_args(argv)
    try:
        gold_path=args.upstream_root/'data'/'evaluation'/'gold_labels'/'dev'/f'gold_admin_{args.lang}.jsonl'
        if not gold_path.is_file():raise ValueError(f'DEV labels unavailable: {gold_path}')
        gold={r['id']:r for r in load_jsonl(gold_path)}
        inp={r['id']:r for r in load_jsonl(args.input_dir/f'dev_{args.lang}.jsonl')}
        e0path=args.e0_checkpoints/f'dev_{args.lang}.checkpoint.jsonl'
        e0={r['id']:r for r in load_jsonl(e0path)} if e0path.exists() else {}
        sig=fingerprint(args,os.environ.get('LLM_NAME',''),os.environ.get('LLM_BASE_URL',''))
        cp=args.e2_checkpoints/f'dev_{args.lang}_e2.checkpoint.jsonl'
        combined=[]
        for rid,row in inp.items():
            if rid not in gold or rid not in e0 or (args.doc_id and rid!=args.doc_id):continue
            ppath=plan_path(args.plan_dir,args.lang,rid)
            if not ppath.exists():continue
            plan=save_or_load_plan(ppath,row,args)
            state=read_document_updates(cp,sig,row,plan)
            pred=completed_document(row,plan,state)
            if pred is not None:combined.append((rid,gold[rid],e0[rid],pred))
        if not combined:
            print('[DEV ONLY] No complete common E0/E2 documents');return 0
        xs=[];ys0=[];ys2=[]
        for rid,g,baseline,candidate in combined:
            for side in ('labels_a','labels_b'):
                if len(baseline[side])!=len(g[side]) or len(candidate[side])!=len(g[side]):
                    raise ValueError(f'Length mismatch {rid}/{side}')
                for v,y0,y2 in zip(g[side],baseline[side],candidate[side]):
                    if v==-1:continue
                    if y0<0 or y2<0:raise ValueError(f'Invalid semantic score for {rid}')
                    xs.append(v);ys0.append(y0);ys2.append(y2)
        print(f'[DEV EXPLORATORY] {args.lang} docs={len(combined)} tokens={len(xs)} '
              f'alignment={args.alignment} prompt={args.prompt}')
        print('E0-PJQ Spearman proxy:',spearman(xs,ys0))
        print('E2 Spearman proxy:',spearman(xs,ys2))
        print('[CAUTION] Non-representative overlap; this is NOT an official UZH score')
        print('IDs:', ', '.join(rowid for rowid,*_ in combined[:15]))
        return 0
    except (ValueError,FileNotFoundError,KeyError,TypeError) as exc:
        print(f'[FAIL CLOSED] {exc}',file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
