"""CLAUSESHIFT — DEV-ONLY offline diagnostics for E0-PJQ vs E1-I.

This script does not make network calls, write checkpoints, or read test data.
Copy into `tools/`, run with `python3 -m tools.diagnose_e1i_quality_dev`.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from src.core import load_jsonl
from src.e1c import chunk_ranges
from src.e1i import read_updates, completed_document, signature
from tools.compare_e0_e1c_dev import spearman


def load_by_id(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing DEV input: {path}")
    records = list(load_jsonl(path))
    result = {r['id']: r for r in records}
    if len(result) != len(records):
        raise ValueError(f"Duplicate document IDs in {path}")
    return result


def triples(gold, e0, e1, side, start=None, end=None):
    arr = []
    n = len(gold[side])
    if not (n == len(e0[side]) == len(e1[side])):
        raise ValueError(f"Length mismatch: {side}")
    start = 0 if start is None else start
    end = n if end is None else end
    for i in range(start, end):
        g, p0, p1 = gold[side][i], e0[side][i], e1[side][i]
        if g == -1: continue
        if p0 == -1 or p1 == -1:
            raise ValueError(f"Semantic token marked punctuation: {side}[{i}]")
        arr.append((i, float(g), float(p0), float(p1)))
    return arr


def brief(title, pairs):
    if not pairs:
        print(f"{title}: no evaluable tokens")
        return
    gold, e0, e1 = zip(*[(g, x, y) for _, g, x, y in pairs])
    a, b = spearman(gold, e0), spearman(gold, e1)
    def fmt(v): return 'undefined' if v is None else f'{v:+.4f}'
    print(f'{title}: tokens={len(pairs)} E0={fmt(a)} E1-I={fmt(b)} '
          f'E1_unique={len(set(e1))} E0_unique={len(set(e0))}')


def main():
    p=argparse.ArgumentParser(description='DEV-only, offline diagnosis; no API and no test reads')
    p.add_argument('--doc-id', default='admin_de_27')
    p.add_argument('--lang', choices=['de','fr','it'], default='de')
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, default=Path('data/inputs'))
    p.add_argument('--e0-checkpoints', type=Path, default=Path('out/checkpoints/e0pjq_publicai_4096'))
    p.add_argument('--e1i-checkpoints', type=Path, default=Path('out/checkpoints/e1i_dev_v1'))
    p.add_argument('--chunk-tokens', type=int, default=24)
    p.add_argument('--context', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--top', type=int, default=8)
    args=p.parse_args()
    model=os.environ.get('LLM_NAME', '')
    endpoint=os.environ.get('LLM_BASE_URL', '')
    if not model or not endpoint:
        raise SystemExit('Set LLM_NAME and LLM_BASE_URL to match the original checkpoint signature.')
    inp=load_by_id(args.input_dir / f'dev_{args.lang}.jsonl')
    gold=load_by_id(args.upstream_root / 'data/evaluation/gold_labels/dev' / f'gold_admin_{args.lang}.jsonl')
    e0=load_by_id(args.e0_checkpoints / f'dev_{args.lang}.checkpoint.jsonl')
    if args.doc_id not in inp or args.doc_id not in gold or args.doc_id not in e0:
        raise SystemExit(f'{args.doc_id} not found in all three DEV sources')
    settings=SimpleNamespace(chunk_tokens=args.chunk_tokens,context=args.context,max_tokens=args.max_tokens)
    sig=signature(settings,model,endpoint)
    state=read_updates(args.e1i_checkpoints / f'dev_{args.lang}_indexed.checkpoint.jsonl',sig,inp,settings)
    row=inp[args.doc_id]
    pred=completed_document(row,state,settings)
    if pred is None:
        raise SystemExit(f'E1-I is incomplete for {args.doc_id}; nothing evaluated')
    g,b,c = gold[args.doc_id],e0[args.doc_id],pred
    ta,tb=row['text_a'].split(),row['text_b'].split()
    print(f'[DEV ONLY] {args.doc_id} ({args.lang}) — offline, no API, no test data')
    all_pairs=[]
    for side in ('labels_a','labels_b'):
        pairs=triples(g,b,c,side)
        brief(side, pairs)
        all_pairs.extend(pairs)
    brief('POOLED', all_pairs)
    windows=chunk_ranges(len(ta),len(tb),args.chunk_tokens)
    for j,(ra,rb) in enumerate(windows,1):
        pairs=triples(g,b,c,'labels_a',*ra)+triples(g,b,c,'labels_b',*rb)
        print(f'chunk {j}/{len(windows)}: EN [{ra[0]}:{ra[1]}] TARGET [{rb[0]}:{rb[1]}]')
        brief('  correlation',pairs)
    print('LARGEST SCORE-ERROR INCREASES: |gold-E1| minus |gold-E0| (illustrative; not Spearman contribution)')
    failures=[]
    for side,tokens in (('labels_a',ta),('labels_b',tb)):
        for i,gv,p0,p1 in triples(g,b,c,side):
            failures.append((abs(gv-p1)-abs(gv-p0),side,i,tokens[i],gv,p0,p1))
    for _,side,i,token,gv,p0,p1 in sorted(failures,reverse=True)[:max(0,args.top)]:
        print(f'  {side}[{i}] {token!r} gold={gv:.2f} E0={p0:.2f} E1-I={p1:.2f}')
    print('[READ ONLY] No checkpoints, gold files, prompts or predictions modified.')

if __name__=='__main__':main()
