"""DEV-only comparison E0 vs completed E2-BATCH; never reads final/test."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from src.core import load_jsonl
from src.e2 import read_document_updates,completed_document
from src.e2_batch import signature,_checkpoint_filename,_plan_filename
from src.e2_alignment import save_or_load_plan
from tools.compare_e0_e1c_dev import spearman


def main(argv=None):
    p=argparse.ArgumentParser(description='DEV-only comparison; official held-out labels prohibited')
    p.add_argument('--lang',required=True,choices=['de','fr','it'])
    p.add_argument('--doc-id',default='')
    p.add_argument('--upstream-root',required=True,type=Path)
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--e0-checkpoints',type=Path,default=Path('out/checkpoints/e0pjq_publicai_4096'))
    p.add_argument('--batch-checkpoints',type=Path,default=Path('out/checkpoints/batch_dev_v114'))
    p.add_argument('--plan-dir',type=Path,default=Path('out/plans/batch_dev_v114'))
    p.add_argument('--alignment',default='semantic_coverage')
    p.add_argument('--prompt',default='fewshot')
    p.add_argument('--embedding-model',default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    p.add_argument('--embedding-revision',default=os.environ.get('EMB_SHA','main'))
    p.add_argument('--chunk-tokens',type=int,default=48)
    p.add_argument('--max-unit-tokens',type=int,default=48)
    p.add_argument('--similarity-threshold',type=float,default=0.46)
    p.add_argument('--merge-penalty',type=float,default=0.06)
    p.add_argument('--coverage-weight',type=float,default=0.50)
    p.add_argument('--anchor-bonus',type=float,default=0.35)
    p.add_argument('--context',type=int,default=20)
    p.add_argument('--max-tokens',type=int,default=1536)
    p.add_argument('--max-scores-per-call',type=int,default=96)
    p.add_argument('--max-units-per-batch',type=int,default=3)
    a=p.parse_args(argv)
    try:
        sig=signature(a,os.environ.get('LLM_NAME',''),os.environ.get('LLM_BASE_URL',''))
        goldpath=a.upstream_root/'data'/'evaluation'/'gold_labels'/'dev'/f'gold_admin_{a.lang}.jsonl'
        gold={r['id']:r for r in load_jsonl(goldpath)}
        inputs={r['id']:r for r in load_jsonl(a.input_dir/f'dev_{a.lang}.jsonl')}
        e0path=a.e0_checkpoints/f'dev_{a.lang}.checkpoint.jsonl'
        e0={r['id']:r for r in load_jsonl(e0path)} if e0path.exists() else {}
        cp=_checkpoint_filename(a.batch_checkpoints,'dev',a.lang)
        pairs=[]
        for rid,row in inputs.items():
            if rid not in e0 or rid not in gold or (a.doc_id and rid!=a.doc_id):continue
            pp=_plan_filename(a.plan_dir,'dev',a.lang,rid)
            if not pp.exists():continue
            plan=save_or_load_plan(pp,row,a)
            st=read_document_updates(cp,sig,row,plan)
            pred=completed_document(row,plan,st)
            if pred is not None:pairs.append((rid,gold[rid],e0[rid],pred))
        if not pairs:
            print('[DEV ONLY] No complete common E0/E2-BATCH documents')
            return 0
        x,y,z=[],[],[]
        for rid,g,p0,pb in pairs:
            for side in ('labels_a','labels_b'):
                if len(g[side])!=len(p0[side]) or len(g[side])!=len(pb[side]):
                    raise ValueError('Token count mismatch')
                for gold_score,a0,ab in zip(g[side],p0[side],pb[side]):
                    if gold_score==-1:continue
                    if a0<0 or ab<0:raise ValueError('Unexpected negative semantic score')
                    x.append(gold_score);y.append(a0);z.append(ab)
        print(f'[BATCH DEV EXPLORATORY] {a.lang} docs={len(pairs)} tokens={len(x)}')
        print('E0-PJQ proxy:',spearman(x,y))
        print('E2-BATCH proxy:',spearman(x,z))
        print('[CAUTION] Only shared complete DEV docs, not the official benchmark')
        print('IDs:', ', '.join(p[0] for p in pairs[:20]))
        return 0
    except (ValueError,FileNotFoundError,KeyError,TypeError) as exc:
        print(f'[BATCH FAIL CLOSED] {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
