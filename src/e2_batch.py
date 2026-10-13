"""CLAUSESHIFT V1.14: multi-unit batched Apertus scoring.

CLI is DEV only. The only path to full/final inputs is src.submission after
signed frozen-manifest preflight. Never uses benchmark gold labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

from .core import LANGS, PredictionError, is_punctuation_only, load_jsonl, validate_input_rows, write_jsonl
from .e1c import context_slice
from .e1i import unique_json, parse_partial, required_indices
from .e2 import FEWSHOT, completed_document, read_document_updates
from .e2_alignment import save_or_load_plan, alignment_summary
from .inference import chat_completion, InferenceError

ROOT=Path(__file__).resolve().parent.parent
VERSION='clauseshift-e2-batch-v114-1'


def signature(args, model, endpoint, split='dev'):
    """Fully isolated from all V1.13 checkpoint signatures."""
    payload={k:getattr(args,k) for k in (
        'alignment','prompt','embedding_model','embedding_revision',
        'chunk_tokens','max_unit_tokens','similarity_threshold','merge_penalty',
        'coverage_weight','anchor_bonus','context','max_tokens',
        'max_scores_per_call','max_units_per_batch')}
    payload.update(version=VERSION,model=model,endpoint=endpoint,split=split,
                   frozen_commit=getattr(args,'frozen_commit','') if split=='final' else '')
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def candidate_batches(row, plan, state, cap, unit_limit):
    """Deterministic non-overlapping requests; each token is requested at most once.

    Greedy first-fit in original unit order. Cap counts requested semantic scores,
    not whitespace tokens. No unit is moved ahead of an earlier unfinished unit.
    """
    if cap < 2 or not 1 <= unit_limit <= 16:
        raise ValueError('Batch limits out of range')
    ta,tb=row['text_a'].split(),row['text_b'].split()
    remaining=[]
    for i,u in enumerate(plan['units']):
        st=state.get(i,{'a':{},'b':{}})
        a=set(required_indices(ta[u['ra'][0]:u['ra'][1]]))-set(st['a'])
        b=set(required_indices(tb[u['rb'][0]:u['rb'][1]]))-set(st['b'])
        if a or b:remaining.append([i,a,b])
    batches=[]
    while remaining:
        batch=[];slots=cap
        for entry in remaining:
            i,a,b=entry
            if not (a or b):continue
            if len(batch)>=unit_limit:break
            # Never send a tiny remainder alone if it will not fit the open batch.
            if slots<2:break
            # A unit can have one requested token; bounded_request supports size >= 2.
            # Avoid splitting a unit unnecessarily when the remaining slot budget is 1.
            from .e2 import bounded_request
            ra,rb=bounded_request(a,b,slots)
            if not ra and not rb:raise ValueError('No progress possible in batch builder')
            batch.append((i,ra,rb));a-=ra;b-=rb
            slots-=len(ra)+len(rb)
            if slots<2:break
        remaining=[rec for rec in remaining if rec[1] or rec[2]]
        if not batch:raise ValueError('Unable to build bounded batch')
        batches.append(batch)
    return batches


def batch_prompt(args, row, plan, batch):
    ta,tb=row['text_a'].split(),row['text_b'].split()
    payload=[]
    for i,req_a,req_b in batch:
        u=plan['units'][i]
        a=context_slice(ta,*u['ra'],args.context)
        b=context_slice(tb,*u['rb'],args.context)
        a['core']=[{'i':j,'token':t} for j,t in enumerate(a['core'])]
        b['core']=[{'i':j,'token':t} for j,t in enumerate(b['core'])]
        payload.append({'unit':i,'english':a,'other_language':b,
                        'requested_a':sorted(req_a),'requested_b':sorted(req_b)})
    base=('Task: evaluate the semantic SIMILARITY of requested CORE tokens to the '
          'corresponding other-language passage. Units are independent consecutive '
          'bilingual windows of one document. Some units have an empty language '
          'side; evaluate additions/omissions from meaning, not from assumed labels. '
          'Use left/right as context only; do not score it. '
          'Indices are zero-based inside each unit core. Score 5=equivalent, '
          '4=very similar, 3=closely related, 2=somewhat related, '
          '1=weakly related, 0=unmatched or different. '
          'Respond with ONLY one raw JSON object of the form '
          '{"units":{"0":{"a":{"0":5},"b":{"0":5}},'
          '"1":{"a":{"0":0},"b":{}}}}. '
          'For every requested unit, output exactly one decimal-string unit key. '
          'Inside each unit, keys a and b map exactly the requested token indices '
          'to numeric similarity scores 0..5, without duplicates, missing indices '
          'or extra indices. Never score punctuation. No markdown, no prose. '
          'Payload:\n'+json.dumps({'units':payload},ensure_ascii=False,separators=(',',':'))
          +'\nReturn JSON only:')
    return ('The next examples illustrate only the similarity SCALE, not the output JSON STRUCTURE of this multi-unit task.\n'+FEWSHOT+'\n'+base) if args.prompt=='fewshot' else base


def parse_batch(text, row, plan, batch):
    obj=unique_json(text)
    if set(obj)!={'units'} or not isinstance(obj['units'],dict):
        raise PredictionError('Batched response must contain only an object units')
    wanted={str(i) for i,_,_ in batch}
    if set(obj['units'])!=wanted:
        raise PredictionError(f'Incorrect batch units: expected {sorted(wanted)}, received {sorted(obj["units"])}')
    ta,tb=row['text_a'].split(),row['text_b'].split()
    results=[]
    for i,req_a,req_b in batch:
        u=plan['units'][i]
        ca=ta[u['ra'][0]:u['ra'][1]]
        cb=tb[u['rb'][0]:u['rb'][1]]
        sa,sb=parse_partial(json.dumps(obj['units'][str(i)]),ca,cb,req_a,req_b)
        # A batch must be atomic and complete. Never save a partial response.
        if set(sa)!=req_a or set(sb)!=req_b:
            raise PredictionError(f'Incomplete batch indices for unit {i}')
        results.append((i,sa,sb))
    return results


def _call_batch(args, row, plan, batch, endpoint, model, key, infer):
    prompt=batch_prompt(args,row,plan,batch)
    start=time.monotonic()
    raw=infer(endpoint,model,prompt,key,timeout=args.timeout,
              max_tokens=args.max_tokens,retries=args.retries)
    data=parse_batch(raw,row,plan,batch)
    return data,round(time.monotonic()-start,2),len(raw)


def _checkpoint_filename(root,split,lang):
    return Path(root)/f'{split}_{lang}_e2batch.checkpoint.jsonl'


def _plan_filename(root,split,lang,doc_id):
    from .e2_alignment import plan_path
    # Validate doc ID exactly as for DEV plans, but separate the namespaces.
    name=plan_path(root,lang,doc_id).name
    return Path(root)/(name if split=='dev' else 'final_'+name)


def _write_full_atomic(dest, predictions):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name(dest.name+'.partial')
    try:
        write_jsonl(tmp,predictions)
        os.replace(tmp,dest)
    finally:
        if tmp.exists():tmp.unlink()


def run_dataset(args, *, split='dev', infer=chat_completion, embedder=None):
    if split not in ('dev','final'):raise ValueError('Unknown split')
    if split=='final' and not getattr(args,'_frozen_authorized',False):
        raise ValueError('Final/test inputs are forbidden without frozen submission preflight')
    if args.alignment not in ('semantic_coverage','semantic','proportional') or args.prompt not in ('short','fewshot'):
        raise ValueError('Unknown method configuration')
    if not (2<=args.max_scores_per_call<=128 and 1<=args.max_units_per_batch<=16
            and 1<=args.workers<=8 and args.launch_interval>=0 and
            args.max_calls>=0 and args.max_documents>=0 and
            args.max_attempts_per_unit>=1 and args.context>=0):
        raise ValueError('Invalid runtime limits')
    if split=='final' and (args.doc_id or args.max_calls or args.max_documents or
                           args.shortest_first or args.plan_only or args.lang!='all'):
        raise ValueError('Final cannot be partial, reordered or plan-only')
    model=os.environ.get('LLM_NAME','')
    endpoint=os.environ.get('LLM_BASE_URL','')
    key=os.environ.get('LLM_API_KEY','')
    if not args.plan_only and ('apertus' not in model.lower() or not endpoint or not key):
        raise ValueError('Apertus model, endpoint and API key required')
    sig=signature(args,model,endpoint,split)
    print(f'[BATCH {split.upper()}] signature={sig[:12]} cap={args.max_scores_per_call} '
          f'units/call<={args.max_units_per_batch} workers={args.workers}',flush=True)
    total_calls=0
    langs=LANGS if args.lang=='all' else (args.lang,)
    for lang in langs:
        rows=list(load_jsonl(Path(args.input_dir)/f'{split}_{lang}.jsonl'))
        validate_input_rows(rows)
        if args.doc_id and not any(r['id']==args.doc_id for r in rows):
            raise ValueError(f'Unknown {split}/{lang} document id {args.doc_id}')
        selected=sorted(rows,key=lambda r:(len(r['text_a'].split())+len(r['text_b'].split()),r['id'])) if args.shortest_first else rows
        cp=_checkpoint_filename(args.checkpoint_dir,split,lang)
        cp.parent.mkdir(parents=True,exist_ok=True)
        count_docs=0
        for row in selected:
            if args.doc_id and row['id']!=args.doc_id:continue
            if args.max_documents and count_docs>=args.max_documents:break
            pl_file=_plan_filename(args.plan_dir,split,lang,row['id'])
            plan=save_or_load_plan(pl_file,row,args,embedder=embedder)
            su=alignment_summary(plan,row)
            print(f'[BATCH PLAN] {lang}/{row["id"]} units={su["units"]} '
                  f'unaligned={su["unaligned_fraction"]:.2%} hash={plan["plan_sha256"][:12]}',flush=True)
            count_docs+=1
            if args.plan_only:continue
            state=read_document_updates(cp,sig,row,plan)
            if completed_document(row,plan,state) is not None:
                print(f'[BATCH DOCUMENT COMPLETE FROM CHECKPOINT] {lang}/{row["id"]}',flush=True)
                continue
            batches=candidate_batches(row,plan,state,args.max_scores_per_call,args.max_units_per_batch)
            unit_attempts={i:state.get(i,{}).get('calls',0) for i in range(len(plan['units']))}
            for batch in batches:
                for i,_,_ in batch:
                    unit_attempts[i]=unit_attempts.get(i,0)+1
                    if unit_attempts[i]>args.max_attempts_per_unit:
                        raise ValueError(f'Unit {i} exceeds per-unit call budget; preserve checkpoint')
            if args.max_calls:
                budget=args.max_calls-total_calls
                if budget<=0:
                    print('[BATCH CALL LIMIT] checkpoint preserved',flush=True)
                    return 0
                batches=batches[:budget]
            # Parallel inference, serial durable checkpoint commits; failure is fail-closed.
            errors=[]
            with cp.open('a',encoding='utf-8') as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures={}
                for batch in batches:
                    print(f'[BATCH START] {lang}/{row["id"]} units={[x[0]+1 for x in batch]} '
                          f'scores={sum(len(a)+len(b) for _,a,b in batch)}',flush=True)
                    f=pool.submit(_call_batch,args,row,plan,batch,endpoint,model,key,infer)
                    futures[f]=batch
                    if args.launch_interval:time.sleep(args.launch_interval)
                for future in as_completed(futures):
                    batch=futures[future]
                    total_calls+=1
                    try:
                        results,elapsed,chars=future.result()
                        lines=[]
                        for i,sa,sb in results:
                            u=plan['units'][i]
                            lines.append(json.dumps({
                                '_signature':sig,'plan_sha256':plan['plan_sha256'],
                                'id':row['id'],'unit_index':i,'ra':u['ra'],'rb':u['rb'],
                                'a':{str(k):v for k,v in sa.items()},
                                'b':{str(k):v for k,v in sb.items()},
                                'seconds':elapsed,'output_chars':chars,
                                'batched':True},ensure_ascii=False))
                        fh.write('\n'.join(lines)+'\n');fh.flush();os.fsync(fh.fileno())
                        print(f'[BATCH SAVED] {lang}/{row["id"]} units={[i+1 for i,_,_ in results]} '
                              f'scores={sum(len(a)+len(b) for _,a,b in batch)} elapsed={elapsed}s',flush=True)
                    except (InferenceError,PredictionError,ValueError) as exc:
                        errors.append(exc)
                        print(f'[BATCH FAIL CLOSED] {lang}/{row["id"]}: {exc}',file=sys.stderr,flush=True)
            if errors:return 1
            state=read_document_updates(cp,sig,row,plan)
            if completed_document(row,plan,state) is not None:
                print(f'[BATCH DOCUMENT COMPLETE] {lang}/{row["id"]}',flush=True)
            else:
                print(f'[BATCH DOCUMENT PARTIAL] {lang}/{row["id"]}; scores safely retained',flush=True)
            if args.max_calls and total_calls>=args.max_calls:
                print(f'[BATCH CALL LIMIT] {total_calls} calls; partial scores retained',flush=True)
                return 0
        if not args.plan_only and not args.doc_id:
            # No partial or reordered file is accepted as a benchmark output.
            complete=[]
            for row in rows:
                pl=_plan_filename(args.plan_dir,split,lang,row['id'])
                if not pl.exists():complete.append(None);continue
                plan=save_or_load_plan(pl,row,args,embedder=embedder)
                state=read_document_updates(cp,sig,row,plan)
                complete.append(completed_document(row,plan,state))
            print(f'[BATCH STATUS] {lang}: {sum(x is not None for x in complete)}/{len(rows)}',flush=True)
            if all(x is not None for x in complete):
                dest=Path(args.output_dir)/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'
                _write_full_atomic(dest,complete)
                print(f'[BATCH FULL EXPORT] {dest}',flush=True)
            elif split=='final':
                raise ValueError(f'Final {lang} incomplete: no prediction file exported')
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description='E2 batch DEV ONLY; final available exclusively via frozen submission runner')
    p.add_argument('--lang',choices=['all',*LANGS],default='de')
    p.add_argument('--doc-id',default='')
    p.add_argument('--input-dir',type=Path,default=ROOT/'data'/'inputs')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'out'/'checkpoints'/'batch_dev_v114')
    p.add_argument('--plan-dir',type=Path,default=ROOT/'out'/'plans'/'batch_dev_v114')
    p.add_argument('--output-dir',type=Path,default=ROOT/'out'/'predictions_batch_dev')
    p.add_argument('--alignment',choices=['semantic_coverage','semantic','proportional'],default='semantic_coverage')
    p.add_argument('--prompt',choices=['fewshot','short'],default='fewshot')
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
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--launch-interval',type=float,default=0)
    p.add_argument('--timeout',type=float,default=75)
    p.add_argument('--retries',type=int,default=0)
    p.add_argument('--max-calls',type=int,default=1)
    p.add_argument('--max-documents',type=int,default=1)
    p.add_argument('--max-attempts-per-unit',type=int,default=8)
    p.add_argument('--shortest-first',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(argv)
    try:return run_dataset(args)
    except (ValueError,FileNotFoundError,RuntimeError,PredictionError) as exc:
        print(f'[BATCH FAIL CLOSED] {exc}',file=sys.stderr,flush=True)
        return 1

if __name__=='__main__':sys.exit(main())
