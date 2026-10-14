"""Read-only DEV-only quality check for CLAUSESHIFT E2-SINGLE completed predictions.

This is exploratory, not the official SwissGov-RSD macro score. Does not call an API,
access held-out test/final data, or write checkpoint/prediction/gold files.
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

from src.core import load_jsonl
from src.e2 import completed_document, fingerprint, read_document_updates
from src.e2_alignment import plan_path, save_or_load_plan
from tools.compare_e0_e1c_dev import spearman


def get_args(argv=None):
    p = argparse.ArgumentParser(description='DEV only: score completed E2-SINGLE checkpoints against DEV gold')
    p.add_argument('--lang', choices=('de','fr','it'), required=True)
    p.add_argument('--doc-id', default='', help='DEV ID; omit to score every completed DEV document in checkpoint')
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, default=Path('data/inputs'))
    p.add_argument('--plan-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--alignment', choices=('semantic_coverage','semantic','proportional'), default='semantic_coverage')
    p.add_argument('--prompt', choices=('fewshot','short'), default='fewshot')
    p.add_argument('--embedding-model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    p.add_argument('--embedding-revision', default=os.environ.get('EMB_SHA',''))
    p.add_argument('--chunk-tokens', type=int, default=48)
    p.add_argument('--max-unit-tokens', type=int, default=48)
    p.add_argument('--similarity-threshold', type=float, default=0.46)
    p.add_argument('--merge-penalty', type=float, default=0.06)
    p.add_argument('--coverage-weight', type=float, default=0.50)
    p.add_argument('--anchor-bonus', type=float, default=0.35)
    p.add_argument('--context', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=1536)
    p.add_argument('--max-scores-per-call', type=int, default=96)
    return p.parse_args(argv)


def evaluate(args):
    model = os.environ.get('LLM_NAME','')
    endpoint = os.environ.get('LLM_BASE_URL','')
    if not model or not endpoint or not args.embedding_revision:
        raise ValueError('LLM_NAME, LLM_BASE_URL, and full EMB_SHA/--embedding-revision are required to verify signed checkpoint')
    inputs = args.input_dir / f'dev_{args.lang}.jsonl'
    gold_path = args.upstream_root / 'data' / 'evaluation' / 'gold_labels' / 'dev' / f'gold_admin_{args.lang}.jsonl'
    cp_path = args.checkpoint_dir / f'dev_{args.lang}_e2.checkpoint.jsonl'
    for p in (inputs, gold_path, cp_path):
        if not p.is_file():
            raise FileNotFoundError(f'DEV input or checkpoint missing: {p}')
    # No access to test/final paths or labels; do not derive outputs from unavailable documents.
    gold = {r['id']:r for r in load_jsonl(gold_path)}
    rows = list(load_jsonl(inputs))
    if args.doc_id and not any(row['id']==args.doc_id for row in rows):
        raise ValueError(f'{args.doc_id} is not present in DEV {args.lang} input')
    signature=fingerprint(args,model,endpoint)
    pairs=[]
    for row in rows:
        doc_id=row['id']
        if args.doc_id and doc_id!=args.doc_id: continue
        p=plan_path(args.plan_dir,args.lang,doc_id)
        if not p.exists():
            if args.doc_id: raise FileNotFoundError(f'Plan missing: {p}')
            continue
        plan=save_or_load_plan(p,row,args)
        state=read_document_updates(cp_path,signature,row,plan)
        pred=completed_document(row,plan,state)
        if pred is None:
            if args.doc_id:raise ValueError(f'{doc_id}: E2 prediction incomplete')
            continue
        if doc_id not in gold:raise ValueError(f'{doc_id}: DEV gold missing')
        pairs.append((doc_id,gold[doc_id],pred))
    if not pairs:raise ValueError('No completed DEV documents with labels were found')
    totals={'labels_a': ([],[]),'labels_b':([],[])}
    for doc_id, gold_row, pred in pairs:
        for side,(truth,guess) in totals.items():
            gl=gold_row[side]
            pr=pred[side]
            if len(gl)!=len(pr):raise ValueError(f'{doc_id}/{side}: size mismatch')
            for g,v in zip(gl,pr):
                if g==-1:
                    if v!=-1:raise ValueError(f'{doc_id}/{side}: punctuation alignment mismatch')
                    continue
                if v<0:raise ValueError(f'{doc_id}/{side}: semantic position marked as punctuation')
                truth.append(float(g));guess.append(float(v))
    pooled_g=totals['labels_a'][0]+totals['labels_b'][0]
    pooled_p=totals['labels_a'][1]+totals['labels_b'][1]
    print(f'[E2 DEV ONLY] lang={args.lang} docs={len(pairs)} signature={signature[:12]} tokens={len(pooled_g)}')
    for side,(g,p) in totals.items():
        distribution=sorted(Counter(g).items())
        print(f'{side}: tokens={len(g)} gold_distribution={distribution} spearman={spearman(g,p)}')
    print(f'POOLED DEV Spearman proxy: {spearman(pooled_g,pooled_p)}')
    print('IDs: '+', '.join(doc_id for doc_id,*_ in pairs[:20]))
    print('[READ ONLY] DEV gold and existing checkpoints only. No final/test/API/exports touched.')
    print('[CAUTION] Exploratory pooled metric, NOT official UZH evaluation, and biased if documents were selected after tuning.')


def main(argv=None):
    try:
        evaluate(get_args(argv))
        return 0
    except (ValueError,FileNotFoundError,KeyError,TypeError) as exc:
        print(f'[DEV FAIL CLOSED] {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
