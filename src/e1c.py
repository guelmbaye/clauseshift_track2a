"""E1-C: DEV-ONLY chunked Apertus numeric-only inference.

Never reads held-out test data. This is NOT the official paper baseline.
Reference-order output is produced only after all development documents pass.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from .core import LANGS, PredictionError, is_punctuation_only, load_jsonl, write_jsonl, valid_prediction, validate_input_rows, parse_json_object
from .inference import chat_completion, InferenceError

ROOT=Path(__file__).resolve().parent.parent
PROMPT_VERSION='clauseshift-e1c-numeric-bidirectional-v1'


def chunk_ranges(n_a, n_b, chunk_tokens):
    if chunk_tokens < 8: raise ValueError('chunk-tokens must be >= 8')
    if n_a < 1 or n_b < 1: raise ValueError('Empty document not supported')
    count=max(1, math.ceil(max(n_a,n_b)/chunk_tokens))
    # Some segments on the shorter side may be empty; preserve index coverage exactly.
    return [((i*n_a//count, (i+1)*n_a//count),
             (i*n_b//count, (i+1)*n_b//count)) for i in range(count)]


def context_slice(tokens, start, stop, halo):
    return {
        'left': tokens[max(0,start-halo):start],
        'core': tokens[start:stop],
        'right': tokens[stop:min(len(tokens),stop+halo)]
    }


def make_prompt(tokens_a,tokens_b,ra,rb,context):
    a=context_slice(tokens_a,*ra,context)
    b=context_slice(tokens_b,*rb,context)
    instructions=(
        'Task: assign semantic SIMILARITY scores to individual core tokens across English '
        'and the other-language text. Consider both directions, including additions and omissions. '
        'The two core windows are paired by approximate document position; alignment may be imperfect. '
        'Read left/right context in both languages for meaning, but output SCORES ONLY for each CORE. '
        'Scale: 5 exact semantic equivalent, 4 very similar, 3 closely related, '
        '2 somewhat similar, 1 weak relation, 0 missing/different information. '
        'For punctuation-only tokens return -1. '
        'Do not copy tokens into the output. '
        'Respond with one JSON object and exactly two keys: scores_a, scores_b. '
        'They must be lists of numbers in original CORE order, one per token, '
        'with exact list lengths. No prose or markdown.\n'
        'Example: CORE_A=["Monday","."]; CORE_B=["vendredi","."] '
        '=> {"scores_a":[0,-1],"scores_b":[0,-1]}\n'
        'JSON input:\n'
    )
    payload={'english':a,'other_language':b,
             'core_length_a':len(a['core']),'core_length_b':len(b['core'])}
    return instructions+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\nReturn JSON:'


def score_arrays(text, tokens_a, tokens_b):
    data=parse_json_object(text)
    if set(data)!= {'scores_a','scores_b'}:
        raise PredictionError(f'E1C unexpected JSON keys: {sorted(data)}')
    output=[]
    for key,tokens in (('scores_a',tokens_a),('scores_b',tokens_b)):
        vals=data[key]
        if not isinstance(vals,list) or len(vals)!=len(tokens):
            raise PredictionError(f'{key}: expected {len(tokens)} scores; got {len(vals) if isinstance(vals,list) else "non-list"}')
        normalized=[]
        for i,(token,val) in enumerate(zip(tokens,vals)):
            if isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val):
                raise PredictionError(f'{key}[{i}] not a finite numeric score')
            if is_punctuation_only(token):
                if val!=-1: raise PredictionError(f'{key}[{i}] punctuation must be -1')
                normalized.append(-1.0)
            else:
                if not 0<=val<=5: raise PredictionError(f'{key}[{i}] similarity score out of range')
                normalized.append(1-float(val)/5)
        output.append(normalized)
    return output[0],output[1]


def fingerprint(args,model,url):
    data={'version':PROMPT_VERSION,'model':model,'endpoint':url,
          'chunk_tokens':args.chunk_tokens,'context':args.context,
          'max_tokens':args.max_tokens,'split':'dev'}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def read_chunks(path,signature):
    rows={}
    if not path.exists():return rows
    for row in load_jsonl(path):
        if row.get('_signature')!=signature:
            raise ValueError('E1-C checkpoint signature changed; choose a NEW checkpoint folder')
        key=(row['id'],row['chunk_index'])
        if key in rows:raise ValueError(f'Duplicate checkpoint {key}')
        rows[key]=row
    return rows


def collect_one(row,found,args):
    ta=row['text_a'].split();tb=row['text_b'].split()
    windows=chunk_ranges(len(ta),len(tb),args.chunk_tokens)
    scores_a=[];scores_b=[]
    for i,(ra,rb) in enumerate(windows):
        saved=found.get((row['id'],i))
        if saved is None:return None
        if saved['ra']!=list(ra) or saved['rb']!=list(rb):
            raise ValueError(f'Stale chunk boundaries: {row["id"]} chunk={i}')
        if saved['input_sha256']!=chunk_hash(ta,tb,ra,rb,args.context):
            raise ValueError(f'Input changed for {row["id"]} chunk={i}')
        expected_a=ra[1]-ra[0];expected_b=rb[1]-rb[0]
        if len(saved['scores_a'])!=expected_a or len(saved['scores_b'])!=expected_b:
            raise ValueError('Invalid saved chunk length')
        scores_a+=saved['scores_a'];scores_b+=saved['scores_b']
    pred={'id':row['id'],'text_a':row['text_a'],'text_b':row['text_b'],
          'labels_a':scores_a,'labels_b':scores_b}
    valid_prediction(pred,row)
    return pred


def chunk_hash(ta,tb,ra,rb,context):
    return hashlib.sha256(json.dumps([ta,tb,list(ra),list(rb),context],ensure_ascii=False).encode()).hexdigest()


def run(args, infer=chat_completion):
    if args.context<0 or args.max_calls<0 or args.max_documents<0:
        raise ValueError('Nonnegative context/max-calls/max-documents required')
    model=os.environ.get('LLM_NAME','')
    url=os.environ.get('LLM_BASE_URL','')
    key=os.environ.get('LLM_API_KEY','')
    if not model or 'apertus' not in model.lower() or not url:
        raise ValueError('Set valid LLM_NAME (Apertus) and LLM_BASE_URL')
    sig=fingerprint(args,model,url)
    print(f'[E1-C DEV ONLY] model={model} signature={sig[:12]} chunk={args.chunk_tokens} context={args.context}',flush=True)
    langs=LANGS if args.lang=='all' else (args.lang,)
    count_api=0
    for lang in langs:
        allrows=list(load_jsonl(Path(args.input_dir)/f'dev_{lang}.jsonl'))
        validate_input_rows(allrows)
        cp=Path(args.checkpoint_dir)/f'dev_{lang}_chunks.checkpoint.jsonl'
        cp.parent.mkdir(parents=True,exist_ok=True)
        found=read_chunks(cp,sig)
        originals={r['id']:r for r in allrows}
        if any(doc not in originals for doc,_ in found):raise ValueError('Unknown ID in chunk checkpoint')
        ordered=sorted(allrows,key=lambda r:(len(r['text_a'].split())+len(r['text_b'].split()),r['id'])) if args.shortest_first else allrows
        completed=0
        with cp.open('a',encoding='utf-8') as handle:
            for row in ordered:
                if args.doc_id and row['id']!=args.doc_id:continue
                if args.max_documents and completed>=args.max_documents:break
                if collect_one(row,found,args) is not None:continue
                ta=row['text_a'].split();tb=row['text_b'].split()
                windows=chunk_ranges(len(ta),len(tb),args.chunk_tokens)
                for idx,(ra,rb) in enumerate(windows):
                    if (row['id'],idx) in found:continue
                    if args.max_calls and count_api>=args.max_calls:
                        print(f'[E1-C CALL LIMIT] {count_api} calls; checkpoints preserved',flush=True)
                        return 0
                    host=urlsplit(url).hostname or '<unknown>'
                    print(f'[E1-C START] {lang}/{row["id"]} chunk={idx+1}/{len(windows)} '
                          f'core_a={ra[1]-ra[0]} core_b={rb[1]-rb[0]} provider={host}',flush=True)
                    prompt=make_prompt(ta,tb,ra,rb,args.context)
                    ts=time.monotonic()
                    try:
                        response=infer(url,model,prompt,key,timeout=args.timeout,
                                       max_tokens=args.max_tokens,retries=args.retries)
                        a,b=score_arrays(response,ta[ra[0]:ra[1]],tb[rb[0]:rb[1]])
                    except (InferenceError,PredictionError,ValueError) as exc:
                        print(f'[E1-C FAIL CLOSED] {lang}/{row["id"]} chunk={idx+1} '
                              f'elapsed={time.monotonic()-ts:.1f}s: {exc}',file=sys.stderr,flush=True)
                        return 1
                    rec={'_signature':sig,'id':row['id'],'chunk_index':idx,
                         'ra':list(ra),'rb':list(rb),'scores_a':a,'scores_b':b,
                         'input_sha256':chunk_hash(ta,tb,ra,rb,args.context),
                         'seconds':round(time.monotonic()-ts,2),
                         'response_chars':len(response)}
                    handle.write(json.dumps(rec,ensure_ascii=False)+'\n');handle.flush()
                    found[(row['id'],idx)]=rec
                    count_api+=1
                    print(f'[E1-C SAVED] {lang}/{row["id"]} chunk={idx+1}/{len(windows)} '
                          f'elapsed={rec["seconds"]}s',flush=True)
                if collect_one(row,found,args) is None:
                    raise ValueError('Internal chunk completion error')
                completed+=1
                print(f'[E1-C DOCUMENT COMPLETE] {lang}/{row["id"]}',flush=True)
        complete=[collect_one(row,found,args) for row in allrows]
        done=sum(p is not None for p in complete)
        print(f'[E1-C STATUS] {lang} {done}/{len(allrows)} documents complete',flush=True)
        if done==len(allrows):
            output=Path(args.output_dir)/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'
            write_jsonl(output,complete)
            print(f'[E1-C FULL DEV EXPORT] {output}',flush=True)
        else:
            print('[E1-C PARTIAL] No official-format output exported',flush=True)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description='E1-C candidate method; strictly DEV only; no test access')
    p.add_argument('--lang',choices=['all',*LANGS],default='de')
    p.add_argument('--input-dir',type=Path,default=ROOT/'data'/'inputs')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'out'/'checkpoints'/'e1c_dev_v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'out'/'predictions_e1c_dev')
    p.add_argument('--chunk-tokens',type=int,default=48)
    p.add_argument('--context',type=int,default=20)
    p.add_argument('--max-tokens',type=int,default=768)
    p.add_argument('--timeout',type=float,default=75)
    p.add_argument('--retries',type=int,default=0)
    p.add_argument('--max-calls',type=int,default=2)
    p.add_argument('--max-documents',type=int,default=1)
    p.add_argument('--shortest-first',action='store_true')
    p.add_argument('--doc-id',default='')
    args=p.parse_args(argv)
    try:
        return run(args)
    except (ValueError,FileNotFoundError) as exc:
        print(f'[E1-C FAIL CLOSED] {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':sys.exit(main())
