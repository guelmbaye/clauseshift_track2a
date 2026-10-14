"""Read-only DEV-only unit-level error audit for CLAUSESHIFT E2-SINGLE.

Loads signed E2 checkpoints and original DEV gold; does not write anything,
call providers, or access final/test inputs. Requires a completed DEV document.
"""
import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.core import load_jsonl
from src.e2 import completed_document, fingerprint, read_document_updates
from src.e2_alignment import plan_path, save_or_load_plan
from tools.compare_e0_e1c_dev import spearman


def parse_args():
    p = argparse.ArgumentParser(description='Read-only E2-SINGLE per-unit DEV error audit')
    p.add_argument('--lang', choices=['de', 'fr', 'it'], required=True)
    p.add_argument('--doc-id', required=True)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, default=Path('data/inputs'))
    p.add_argument('--plan-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--alignment', default='semantic_coverage')
    p.add_argument('--prompt', default='fewshot')
    p.add_argument('--embedding-model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    p.add_argument('--embedding-revision', default=os.environ.get('EMB_SHA', ''))
    p.add_argument('--chunk-tokens', type=int, default=48)
    p.add_argument('--max-unit-tokens', type=int, default=48)
    p.add_argument('--similarity-threshold', type=float, default=0.46)
    p.add_argument('--merge-penalty', type=float, default=0.06)
    p.add_argument('--coverage-weight', type=float, default=0.50)
    p.add_argument('--anchor-bonus', type=float, default=0.35)
    p.add_argument('--context', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=1536)
    p.add_argument('--max-scores-per-call', type=int, default=96)
    return p.parse_args()


def distribution(values):
    return sorted((float(k), v) for k,v in Counter(round(float(s),3) for s in values).items())


def main():
    args = parse_args()
    if not args.doc_id.startswith('admin_' + args.lang + '_'):
        raise ValueError('ID must match the requested DEV language')
    if not args.embedding_revision or not os.environ.get('LLM_NAME') or not os.environ.get('LLM_BASE_URL'):
        raise ValueError('EMB_SHA, LLM_NAME and LLM_BASE_URL required for signed-checkpoint verification')
    dev_input = args.input_dir / f'dev_{args.lang}.jsonl'
    dev_gold = args.upstream_root / 'data/evaluation/gold_labels/dev' / f'gold_admin_{args.lang}.jsonl'
    row = next((r for r in load_jsonl(dev_input) if r['id'] == args.doc_id), None)
    gold = next((r for r in load_jsonl(dev_gold) if r['id'] == args.doc_id), None)
    if row is None or gold is None: raise ValueError('DEV document or DEV gold not found')
    plan = save_or_load_plan(plan_path(args.plan_dir,args.lang,args.doc_id),row,args)
    sig = fingerprint(args,os.environ['LLM_NAME'],os.environ['LLM_BASE_URL'])
    cp = args.checkpoint_dir / f'dev_{args.lang}_e2.checkpoint.jsonl'
    state = read_document_updates(cp,sig,row,plan)
    pred = completed_document(row,plan,state)
    if pred is None: raise ValueError('Document incomplete; cannot audit reliably')
    groups = defaultdict(lambda: ([],[]))
    print(f'[DEV E2 UNIT AUDIT] lang={args.lang} id={args.doc_id} signature={sig[:12]} units={len(plan["units"])}')
    total_gold, total_pred = [], []
    for number,u in enumerate(plan['units'],1):
        gvals, pvals, mistakes = [], [], []
        for side,rng in (('a',u['ra']),('b',u['rb'])):
            a,b=rng
            words=row['text_'+side].split()[a:b]
            gl=gold['labels_'+side][a:b]
            pr=pred['labels_'+side][a:b]
            if len(words)!=len(gl) or len(gl)!=len(pr):raise ValueError(f'Invalid lengths on unit {number}')
            for offset,(word,g,p) in enumerate(zip(words,gl,pr)):
                if g==-1:
                    if p!=-1: raise ValueError('Punctuation disagreement')
                    continue
                if p<0 or not math.isfinite(p):raise ValueError('Invalid semantic prediction')
                g,p=float(g),float(p)
                gvals.append(g);pvals.append(p)
                total_gold.append(g);total_pred.append(p)
                groups[u['kind']][0].append(g);groups[u['kind']][1].append(p)
                mistakes.append((abs(g-p), side, a+offset, word, g, p))
        corr=spearman(gvals,pvals)
        mae=sum(abs(a-b) for a,b in zip(gvals,pvals))/len(gvals) if gvals else None
        print(f'UNIT {number}/{len(plan["units"])} kind={u["kind"]} evaluated={len(gvals)} spearman={corr} mae={mae:.3f}' if mae is not None else f'UNIT {number} no scored tokens')
        print(f'  gold={distribution(gvals)} predicted={distribution(pvals)}')
        for err,side,index,word,g,p in sorted(mistakes,reverse=True)[:3]:
            print(f'  mismatch {side}[{index}] {word!r}: gold={g:.2f} predicted={p:.2f} absolute_error={err:.2f}')
    print(f'POOLED: tokens={len(total_gold)} spearman={spearman(total_gold,total_pred)}')
    for kind,(g,p) in sorted(groups.items()):
        mae=sum(abs(a-b) for a,b in zip(g,p))/len(g) if g else None
        print(f'BY_KIND {kind}: tokens={len(g)} spearman={spearman(g,p)} mae={mae:.3f}' if mae is not None else f'BY_KIND {kind}: no scored tokens')
    print('[READ ONLY] DEV input/gold and signed checkpoints only; no API/final/test/exports')
    print('[CAUTION] Descriptive errors, not an official benchmark or a basis for test-data tuning')


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError,FileNotFoundError) as e:
        print('[DEV AUDIT FAIL CLOSED]',str(e),file=sys.stderr)
        sys.exit(1)
