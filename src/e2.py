"""CLAUSESHIFT E2 DEV-only controlled ablation runner.

E2-A: semantic alignment + short indexed prompt
E2-B: proportional alignment + paper-example-adapted indexed prompt
E2-C: semantic alignment + paper-example-adapted indexed prompt
No held-out test access. No invented semantic scores; full coverage required.
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

from .core import LANGS, PredictionError, is_punctuation_only, load_jsonl, validate_input_rows, valid_prediction, write_jsonl
from .e1i import encode_indexed_prompt, parse_partial, required_indices
from .e1c import context_slice
from .e2_alignment import alignment_summary, plan_path, save_or_load_plan
from .inference import chat_completion, InferenceError

ROOT=Path(__file__).resolve().parent.parent
VERSION='clauseshift-e2-aligned-indexed-v1'

# Score/label examples are from appendix E, reformatted into indexed output. This is
# an ADAPTATION, not the exact baseline prompt. Do not claim otherwise.
FEWSHOT = '''Here are three training-format examples of the semantic similarity scale.
Example 1:
Input A: ["Iran", "hopes", "nuclear", "talks", "will", "yield", "ˋ", "roadmap", "’"]
Input B: ["Iran", "Nuclear", "Talks", "in", "Geneva", "Spur", "High", "Hopes"]
Output: {"a":{"0":5,"1":4,"2":5,"3":5,"4":3,"5":3,"7":2},"b":{"0":5,"1":5,"2":5,"3":0,"4":0,"5":3,"6":2,"7":2}}
Example 2:
Input A: ["Books", "To", "Help", "Kids", "Talk", "About", "Boston", "Marathon", "News"]
Input B: ["Report", "of", "2", "explosions", "at", "finish", "line", "of", "Boston", "Marathon"]
Output: {"a":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":4,"6":4,"7":4,"8":4},"b":{"0":1,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":4,"8":4,"9":4}}
Example 3:
Input A: ["Chinese", "shares", "close", "lower", "Wednesday"]
Input B: ["Chinese", "shares", "close", "higher", "Friday"]
Output: {"a":{"0":5,"1":5,"2":5,"3":0,"4":3},"b":{"0":5,"1":5,"2":5,"3":0,"4":3}}
Punctuation-only tokens are never requested. Score 5=equivalent, 0=unrelated.
'''


def scoring_prompt(args,ta,tb,ra,rb,need_a,need_b):
    base=encode_indexed_prompt(ta,tb,ra,rb,args.context,need_a,need_b)
    if args.prompt=='short':return base
    if args.prompt=='fewshot':return FEWSHOT+'\n'+base
    raise ValueError('Unknown prompt variant')


def fingerprint(args,model,url):
    data={'version':VERSION,'model':model,'endpoint':url,'prompt':args.prompt,
          'context':args.context,'max_tokens':args.max_tokens,'split':'dev',
          'alignment':args.alignment,'chunk_tokens':args.chunk_tokens,
          'embedding_model':args.embedding_model,'embedding_revision':args.embedding_revision,
          'max_unit_tokens':args.max_unit_tokens,
          'similarity_threshold':args.similarity_threshold,'merge_penalty':args.merge_penalty}
    if getattr(args,'max_scores_per_call',0):
        data['max_scores_per_call']=args.max_scores_per_call
    if args.alignment=='semantic_coverage':
        data.update(coverage_weight=args.coverage_weight,anchor_bonus=args.anchor_bonus)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()



def bounded_request(missing_a, missing_b, limit):
    """Select an exact, deterministic subset of nonpunctuation indices for one call.

    Does not rescore previously saved indices. A request can be empty on one side,
    but never on both. Zero means no partitioning (the V1.10 behavior).
    """
    a, b = sorted(missing_a), sorted(missing_b)
    if limit == 0 or len(a)+len(b) <= limit:
        return set(a), set(b)
    if limit < 2:
        raise ValueError('Bounded request size must be at least 2')
    # Fair allocation when both sides have work; avoid starving one language.
    target_a = min(len(a), max(1, limit//2)) if a else 0
    target_b = min(len(b), max(1, limit-target_a)) if b else 0
    slots=limit-target_a-target_b
    if slots>0:
        grow_a=min(slots,len(a)-target_a)
        target_a+=grow_a
        slots-=grow_a
    if slots>0:
        target_b+=min(slots,len(b)-target_b)
    return set(a[:target_a]),set(b[:target_b])


def read_updates(path,sig,row,plan):
    state={}
    if not Path(path).exists():return state
    ta,tb=row['text_a'].split(),row['text_b'].split()
    for rec in load_jsonl(path):
        if rec.get('_signature')!=sig or rec.get('plan_sha256')!=plan['plan_sha256']:
            raise ValueError('Incompatible signed checkpoint or alignment plan')
        if rec.get('id')!=row['id']:raise ValueError('Unexpected doc ID in E2 checkpoint')
        idx=rec.get('unit_index')
        if not isinstance(idx,int) or idx<0 or idx>=len(plan['units']):
            raise ValueError('Invalid checkpoint unit index')
        unit=plan['units'][idx]
        if rec.get('ra')!=unit['ra'] or rec.get('rb')!=unit['rb']:
            raise ValueError('Changed checkpoint spans')
        partial=state.setdefault(idx,{'a':{},'b':{},'calls':0})
        for side,tokens,ranges in (('a',ta,unit['ra']),('b',tb,unit['rb'])):
            required=required_indices(tokens[ranges[0]:ranges[1]])
            saved=rec.get(side)
            if not isinstance(saved,dict):raise ValueError('Malformed checkpoint scores')
            for k,value in saved.items():
                if not isinstance(k,str) or not k.isdecimal() or str(int(k))!=k:
                    raise ValueError('Invalid saved index')
                j=int(k)
                if (j not in required or isinstance(value,bool) or
                    not isinstance(value,(int,float)) or not math.isfinite(value) or
                    not 0<=value<=5 or j in partial[side]):
                    raise ValueError('Invalid/duplicated score in signed checkpoint')
                partial[side][j]=float(value)
        partial['calls']+=1
    return state


def completed_document(row,plan,state):
    ta,tb=row['text_a'].split(),row['text_b'].split()
    outs={'a':[],'b':[]}
    for i,unit in enumerate(plan['units']):
        saved=state.get(i,{'a':{},'b':{}})
        for side,source,rr in (('a',ta,unit['ra']),('b',tb,unit['rb'])):
            core=source[rr[0]:rr[1]]
            if set(saved[side])!=required_indices(core):return None
            outs[side].extend(-1.0 if is_punctuation_only(t) else 1-saved[side][j]/5
                              for j,t in enumerate(core))
    pred={'id':row['id'],'text_a':row['text_a'],'text_b':row['text_b'],
          'labels_a':outs['a'],'labels_b':outs['b']}
    valid_prediction(pred,row)
    return pred


def run(args,infer=chat_completion,embedder=None):
    if args.context<0 or args.max_calls<0 or args.max_documents<0 or args.max_attempts_per_unit<1:
        raise ValueError('Invalid context, max-calls, max-documents or attempts')
    if getattr(args,'max_scores_per_call',0) not in (0, *range(2,129)):
        raise ValueError('max-scores-per-call must be 0 or 2..128')
    if args.prompt not in ('short','fewshot') or args.alignment not in ('semantic','semantic_coverage','proportional'):
        raise ValueError('Unsupported experiment configuration')
    if not 0<=args.similarity_threshold<=1 or not 0<=args.merge_penalty<1:
        raise ValueError('Invalid alignment thresholds')
    if args.alignment=='semantic_coverage' and not (0 < args.coverage_weight <= 1 and 0 <= args.anchor_bonus <= 1):
        raise ValueError('Invalid coverage/anchor rewards')
    model=os.environ.get('LLM_NAME','')
    endpoint=os.environ.get('LLM_BASE_URL','')
    key=os.environ.get('LLM_API_KEY','')
    if not args.plan_only and ('apertus' not in model.lower() or not endpoint):
        raise ValueError('Set Apertus LLM_NAME and LLM_BASE_URL')
    sig=fingerprint(args,model,endpoint)
    host=urlsplit(endpoint).hostname or '<none>'
    print(f'[E2 DEV ONLY] alignment={args.alignment} prompt={args.prompt} model={model or "not-needed-for-plan"} signature={sig[:12]}',flush=True)
    calls=0
    langs=LANGS if args.lang=='all' else (args.lang,)
    for lang in langs:
        path=Path(args.input_dir)/f'dev_{lang}.jsonl'
        allrows=list(load_jsonl(path));validate_input_rows(allrows)
        if args.doc_id and not any(r['id']==args.doc_id for r in allrows):
            raise ValueError(f'{args.doc_id} is not a DEV {lang} document')
        cp=Path(args.checkpoint_dir)/f'dev_{lang}_e2.checkpoint.jsonl'
        cp.parent.mkdir(parents=True,exist_ok=True)
        ordered=(sorted(allrows,key=lambda r:(len(r['text_a'].split())+len(r['text_b'].split()),r['id']))
                 if args.shortest_first else allrows)
        count_done=0
        # Each document plan is created offline, then frozen and checksum verified on resume.
        with cp.open('a',encoding='utf-8') as handle:
            for row in ordered:
                if args.doc_id and row['id']!=args.doc_id:continue
                if args.max_documents and count_done>=args.max_documents:break
                pl_path=plan_path(args.plan_dir,lang,row['id'])
                plan=save_or_load_plan(pl_path,row,args,embedder=embedder)
                summary=alignment_summary(plan,row)
                print(f'[E2 PLAN] {lang}/{row["id"]} units={summary["units"]} '
                      f'unaligned_fraction={summary["unaligned_fraction"]:.2%} hash={plan["plan_sha256"][:12]}',flush=True)
                if args.plan_only:
                    print(f'[E2 PLAN ONLY] {pl_path}',flush=True)
                    count_done+=1
                    continue
                # Important: do not derive a high semantic score for any unaligned block.
                state=read_document_updates(cp,sig,row,plan)
                if completed_document(row,plan,state) is not None:continue
                ta,tb=row['text_a'].split(),row['text_b'].split()
                for unit_i,unit in enumerate(plan['units']):
                    ra,rb=unit['ra'],unit['rb']
                    core_a,core_b=ta[ra[0]:ra[1]],tb[rb[0]:rb[1]]
                    required_a,required_b=required_indices(core_a),required_indices(core_b)
                    partial=state.setdefault(unit_i,{'a':{},'b':{},'calls':0})
                    while True:
                        missing_a=required_a-set(partial['a'])
                        missing_b=required_b-set(partial['b'])
                        if not(missing_a or missing_b):
                            print(f'[E2 UNIT COMPLETE] {lang}/{row["id"]} {unit_i+1}/{len(plan["units"])}',flush=True)
                            break
                        if partial['calls']>=args.max_attempts_per_unit:
                            raise ValueError(f'{row["id"]} unit {unit_i}: too many calls; missing_a={sorted(missing_a)} missing_b={sorted(missing_b)}')
                        if args.max_calls and calls>=args.max_calls:
                            print(f'[E2 CALL LIMIT] {calls} paid calls, valid partial scores retained',flush=True)
                            return 0
                        request_a,request_b=bounded_request(
                            missing_a,missing_b,getattr(args,'max_scores_per_call',0))
                        prompt=scoring_prompt(args,ta,tb,tuple(ra),tuple(rb),request_a,request_b)
                        print(f'[E2 START] {lang}/{row["id"]} unit={unit_i+1}/{len(plan["units"])} '
                              f'kind={unit["kind"]} request_a={len(request_a)} request_b={len(request_b)} '
                              f'remaining_before={len(missing_a)+len(missing_b)} provider={host}',flush=True)
                        start=time.monotonic();text=None
                        try:
                            text=infer(endpoint,model,prompt,key,timeout=args.timeout,
                                       max_tokens=args.max_tokens,retries=args.retries)
                            sa,sb=parse_partial(text,core_a,core_b,request_a,request_b)
                        except (InferenceError,PredictionError,ValueError) as exc:
                            if text is not None:
                                raw_sha=hashlib.sha256(text.encode()).hexdigest()[:12]
                                dest=Path(args.diagnostics_dir)/(f'dev_{lang}_{row["id"]}_e2_{unit_i}_{time.time_ns()}_{raw_sha}.txt')
                                dest.parent.mkdir(parents=True,exist_ok=True)
                                dest.write_text(text,encoding='utf-8')
                                print(f'[E2 RAW INVALID SAVED] {dest}',file=sys.stderr,flush=True)
                            print(f'[E2 FAIL CLOSED] {lang}/{row["id"]} unit={unit_i+1} elapsed={time.monotonic()-start:.1f}s: {exc}',file=sys.stderr,flush=True)
                            return 1
                        rec={'_signature':sig,'plan_sha256':plan['plan_sha256'],'id':row['id'],
                            'unit_index':unit_i,'ra':ra,'rb':rb,
                            'a':{str(k):v for k,v in sa.items()},'b':{str(k):v for k,v in sb.items()},
                            'seconds':round(time.monotonic()-start,2),'output_chars':len(text)}
                        handle.write(json.dumps(rec,ensure_ascii=False)+'\n');handle.flush()
                        partial['a'].update(sa);partial['b'].update(sb);partial['calls']+=1;calls+=1
                        print(f'[E2 SAVED] {lang}/{row["id"]} unit={unit_i+1} '
                              f'new_a={len(sa)} new_b={len(sb)} remaining_a={len(missing_a)-len(sa)} '
                              f'remaining_b={len(missing_b)-len(sb)} elapsed={rec["seconds"]}s',flush=True)
                if completed_document(row,plan,state) is None:raise ValueError('Internal partial-coverage error')
                count_done+=1
                print(f'[E2 DOCUMENT COMPLETE] {lang}/{row["id"]}',flush=True)
        if not args.plan_only and not args.doc_id:
            # Never export partial predictions as a complete benchmark submission.
            completed=[]
            for row in allrows:
                pl=plan_path(args.plan_dir,lang,row['id'])
                if not pl.exists():
                    completed.append(None);continue
                plan=save_or_load_plan(pl,row,args,embedder=embedder)
                state=read_document_updates(cp,sig,row,plan)
                completed.append(completed_document(row,plan,state))
            print(f'[E2 STATUS] {lang}: {sum(x is not None for x in completed)}/{len(allrows)} DEV complete',flush=True)
            if all(x is not None for x in completed):
                dst=Path(args.output_dir)/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'
                write_jsonl(dst,completed)
                print(f'[E2 FULL DEV EXPORT] {dst}',flush=True)
    return 0


def read_document_updates(path,sig,row,plan):
    """Read a shared multilingual DEV checkpoint while filtering to one document."""
    if not path.exists():return {}
    state={}
    # Reuse validation on a per-document restricted source, without writing temp files.
    a,b=row['text_a'].split(),row['text_b'].split()
    for rec in load_jsonl(path):
        if rec.get('_signature')!=sig:raise ValueError('Checkpoint signature mismatch')
        if rec.get('id')!=row['id']:continue
        if rec.get('plan_sha256')!=plan['plan_sha256']:raise ValueError('Plan mismatch')
        idx=rec.get('unit_index')
        if type(idx) is not int or not 0<=idx<len(plan['units']):raise ValueError('Bad unit index')
        u=plan['units'][idx]
        if rec.get('ra')!=u['ra'] or rec.get('rb')!=u['rb']:raise ValueError('Changed unit ranges')
        st=state.setdefault(idx,{'a':{},'b':{},'calls':0})
        for side,tokens,rr in (('a',a,u['ra']),('b',b,u['rb'])):
            vals=rec.get(side)
            if not isinstance(vals,dict):raise ValueError('Bad saved map')
            req=required_indices(tokens[rr[0]:rr[1]])
            for k,v in vals.items():
                if (type(k) is not str or not k.isdecimal() or str(int(k))!=k or
                    int(k) not in req or int(k) in st[side] or type(v) not in (float,int) or
                    not math.isfinite(v) or not 0<=v<=5):
                    raise ValueError('Invalid or conflicting checkpoint index')
                st[side][int(k)]=float(v)
        st['calls']+=1
    return state


def main(argv=None):
    p=argparse.ArgumentParser(description='CLAUSESHIFT E2, strictly DEV-only; no held-out test access')
    p.add_argument('--lang',choices=['all',*LANGS],default='de')
    p.add_argument('--doc-id',default='')
    p.add_argument('--input-dir',type=Path,default=ROOT/'data'/'inputs')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'out'/'checkpoints'/'e2a_dev_v1')
    p.add_argument('--plan-dir',type=Path,default=ROOT/'out'/'plans'/'e2a_dev_v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'out'/'predictions_e2_dev')
    p.add_argument('--diagnostics-dir',type=Path,default=ROOT/'out'/'diagnostics')
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
    p.add_argument('--max-scores-per-call',type=int,default=0,
                   help='DEV-only deterministic bounded requests; 0 preserves V1.10 behavior')
    p.add_argument('--timeout',type=float,default=75)
    p.add_argument('--retries',type=int,default=0)
    p.add_argument('--max-calls',type=int,default=1)
    p.add_argument('--max-documents',type=int,default=1)
    p.add_argument('--max-attempts-per-unit',type=int,default=3)
    p.add_argument('--shortest-first',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(argv)
    try:return run(args)
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        print(f'[E2 FAIL CLOSED] {exc}',file=sys.stderr,flush=True);return 1

if __name__=='__main__':sys.exit(main())
