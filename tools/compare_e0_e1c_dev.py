"""Exploratory DEV-only Spearman on matched completed IDs; NOT official UZH scoring."""
import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from src.core import load_jsonl
from src.e1c import collect_one, fingerprint, read_chunks


def ranks(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    out=[0.0]*len(values)
    i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        avg=(i+1+j)/2  # average of 1-based ranks i+1 ... j
        for k in order[i:j]:out[k]=avg
        i=j
    return out


def spearman(xs,ys):
    if len(xs)!=len(ys) or len(xs)<2: return None
    a=ranks(xs);b=ranks(ys)
    ma=sum(a)/len(a);mb=sum(b)/len(b)
    da=math.sqrt(sum((v-ma)**2 for v in a));db=math.sqrt(sum((v-mb)**2 for v in b))
    if da==0 or db==0:return None
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/(da*db)


def main(argv=None):
    p=argparse.ArgumentParser(description='Exploratory DEV-only overlap: E0-PJQ vs E1-C')
    p.add_argument('--lang',choices=['de','fr','it'],required=True)
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,default=Path('data/inputs'))
    p.add_argument('--e0-checkpoints',type=Path,default=Path('out/checkpoints/e0pjq_publicai_4096'))
    p.add_argument('--e1c-checkpoints',type=Path,default=Path('out/checkpoints/e1c_dev_v1'))
    p.add_argument('--chunk-tokens',type=int,default=48)
    p.add_argument('--context',type=int,default=20)
    p.add_argument('--max-tokens',type=int,default=768)
    args=p.parse_args(argv)
    try:
        gold_path=args.upstream_root/'data'/'evaluation'/'gold_labels'/'dev'/f'gold_admin_{args.lang}.jsonl'
        if not gold_path.is_file():raise ValueError(f'Official DEV gold not found at {gold_path}')
        gold={r['id']:r for r in load_jsonl(gold_path)}
        inp={r['id']:r for r in load_jsonl(args.input_dir/f'dev_{args.lang}.jsonl')}
        e0_path=args.e0_checkpoints/f'dev_{args.lang}.checkpoint.jsonl'
        e0={r['id']:r for r in load_jsonl(e0_path)} if e0_path.exists() else {}
        model=os.environ.get('LLM_NAME','');url=os.environ.get('LLM_BASE_URL','')
        if not model or not url:raise ValueError('LLM_NAME and LLM_BASE_URL required for E1-C signature verification')
        settings=SimpleNamespace(chunk_tokens=args.chunk_tokens,context=args.context,max_tokens=args.max_tokens)
        sig=fingerprint(settings,model,url)
        cp=read_chunks(args.e1c_checkpoints/f'dev_{args.lang}_chunks.checkpoint.jsonl',sig)
        paired=[]
        for rowid,row in inp.items():
            if rowid not in e0 or rowid not in gold:continue
            e1=collect_one(row,cp,settings)
            if e1 is not None:paired.append((rowid,gold[rowid],e0[rowid],e1))
        if not paired:
            print('[DEV ONLY] No common completed documents; no comparison yet')
            return 0
        xs=[];ys0=[];ys1=[]
        for rowid,g,b,c in paired:
            for field in ('labels_a','labels_b'):
                if any(len(d[field])!=len(g[field]) for d in (b,c)):
                    raise ValueError(f'{rowid}/{field}: label length mismatch')
                for label,v0,v1 in zip(g[field],b[field],c[field]):
                    if label==-1:continue
                    if v0 == -1 or v1 == -1:
                        raise ValueError(f'{rowid}: semantic-token prediction marked punctuation')
                    xs.append(label);ys0.append(v0);ys1.append(v1)
        print(f'[DEV EXPLORATORY] {args.lang}: matched_documents={len(paired)} tokens={len(xs)}')
        print('[CAUTION] Overlap-only estimate; NOT official UZH dev or test score.')
        print(f'E0-PJQ Spearman proxy: {spearman(xs,ys0)}')
        print(f'E1-C   Spearman proxy: {spearman(xs,ys1)}')
        print('ID sample: '+', '.join(rowid for rowid,*_ in paired[:8]))
        return 0
    except (ValueError,KeyError,FileNotFoundError) as exc:
        print(f'[FAIL CLOSED] {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
