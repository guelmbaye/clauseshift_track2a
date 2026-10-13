"""Split-safe CLI. Only dev is available without explicit final freeze confirmation."""
import argparse, hashlib, json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from urllib.request import urlopen
from pathlib import Path
from urllib.parse import urlsplit
from .core import (LANGS,PAPER_STYLE,encode_prompt,normalize_prediction,valid_prediction,
                   validate_input_rows,load_jsonl,write_jsonl,PredictionError)
from .inference import chat_completion,InferenceError

BASE=Path(__file__).resolve().parent.parent

def path_dev(upstream,lang):
    return Path(upstream)/'data'/'evaluation'/'gold_labels'/'dev'/f'gold_admin_{lang}.jsonl'
def path_full(upstream,lang):
    return Path(upstream)/'data'/'evaluation'/'gold_labels'/'full'/f'gold_admin_{lang}.jsonl'
def path_in(inputdir,split,lang):return Path(inputdir)/f'{split}_{lang}.jsonl'
def path_out(outdir,lang):return Path(outdir)/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'

def do_fetch_dev(args):
    """Fetch only DEV gold files from official upstream, immediately strip gold labels."""
    sha=args.upstream_sha
    if not (sha=='master' or (len(sha)>=7 and all(c in '0123456789abcdef' for c in sha.lower()))):
        raise ValueError('Expected upstream SHA or master')
    for lang in LANGS:
        url=(f'https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/{sha}/'
             f'data/evaluation/gold_labels/dev/gold_admin_{lang}.jsonl')
        with urlopen(url,timeout=30) as response:
            lines=response.read().decode('utf-8').splitlines()
        # A deterministic loader, never download or inspect test or full for development.
        rows=[]
        for line in lines:
            if line.strip():
                r=json.loads(line)
                rows.append({k:r[k] for k in ('id','text_a','text_b')})
        validate_input_rows(rows)
        out=path_in(args.input_dir,'dev',lang)
        write_jsonl(out,rows)
        print(f'[FETCH-DEV] {lang}: {len(rows)} unlabeled inputs -> {out}')

def do_fetch_final(args):
    """Download ONLY full-order text after a deliberate freeze; discard gold labels."""
    if not args.confirm_final or not args.frozen_commit or len(args.frozen_commit)<7:
        raise ValueError('FINAL fetch denied without --confirm-final + --frozen-commit')
    sha=args.upstream_sha
    if not (sha=='master' or (len(sha)>=7 and all(c in '0123456789abcdef' for c in sha.lower()))):
        raise ValueError('Expected upstream SHA or master')
    for lang in LANGS:
        url=(f'https://raw.githubusercontent.com/ZurichNLP/SwissGov-RSD/{sha}/'
             f'data/evaluation/gold_labels/full/gold_admin_{lang}.jsonl')
        with urlopen(url,timeout=45) as response:
            lines=response.read().decode('utf-8').splitlines()
        rows=[]
        for line in lines:
            if line.strip():
                r=json.loads(line)
                rows.append({k:r[k] for k in ('id','text_a','text_b')})
        validate_input_rows(rows)
        out=path_in(args.input_dir,'final',lang)
        write_jsonl(out,rows)
        print(f'[FINAL AFTER FREEZE] {lang}: {len(rows)} input-only full-order entries -> {out}')

def do_prepare(args):
    if args.split=='final':
        if not args.confirm_final or not args.frozen_commit or len(args.frozen_commit)<7:
            raise ValueError('FINAL ONLY: require --confirm-final and --frozen-commit=<Git SHA>')
    for lang in LANGS:
        p=path_full(args.upstream_root,lang) if args.split=='final' else path_dev(args.upstream_root,lang)
        if not p.is_file():raise FileNotFoundError(f'Missing upstream dataset: {p}')
        rows=[]
        # Only emit unlabeled id/text fields. Never use test labels for training or evaluation.
        for r in load_jsonl(p):rows.append({k:r[k] for k in ('id','text_a','text_b')})
        validate_input_rows(rows)
        out=path_in(args.input_dir,args.split,lang)
        write_jsonl(out,rows)
        print(f'[PREPARE {args.split.upper()}] {lang}: {len(rows)} input-only entries -> {out}')
    if args.split=='final':print('[FINAL] Test inputs materialized AFTER FREEZE; do not tune from these.')

def digest_settings(model,prompt,split,max_tokens,base_url,frozen_commit=None):
    return hashlib.sha256(json.dumps([model,prompt,split,max_tokens,base_url,frozen_commit],ensure_ascii=False).encode()).hexdigest()

def read_checkpoint(path,signature):
    rows={}
    if not path.exists():return rows
    for r in load_jsonl(path):
        if r.get('_signature')!=signature:raise ValueError('Checkpoint settings differ. Use new --checkpoint-dir.')
        if r['id'] in rows:raise ValueError(f'Duplicate checkpoint id {r["id"]}')
        rows[r['id']]=r
    return rows

def preserve_invalid_dev_response(lang, row_id, text, error, root=BASE/'out'/'diagnostics'):
    """Persist a failed, completed DEV generation. Never inspect or persist TEST here.

    Pure diagnostic: does not change scoring, the method signature, or checkpoints.
    """
    if not re.fullmatch(r'(de|fr|it)', lang):
        raise ValueError('Refusing unknown language for diagnostic')
    directory=Path(root)
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(directory,0o700)
    clean_id=re.sub(r'[^A-Za-z0-9_-]+','_',row_id)[:80]
    sha=hashlib.sha256(text.encode('utf-8')).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    filename=directory/f'dev_{lang}_{clean_id}_{stamp}_{sha[:10]}_raw.txt'
    fd=os.open(filename,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as f: f.write(text)
    print(f'[RAW INVALID SAVED] dev/{lang}/{row_id} -> {filename} sha256={sha}', flush=True)
    cause=error.__cause__
    if isinstance(cause,json.JSONDecodeError):
        start=max(0,cause.pos-85); end=min(len(text),cause.pos+85)
        print(f'[JSON SYNTAX] line={cause.lineno} column={cause.colno} char={cause.pos}; '
              f'context={text[start:end]!r}',flush=True)
    return filename

def do_run(args):
    if args.split=='final' and (not args.confirm_final or not args.frozen_commit or len(args.frozen_commit)<7):
        raise ValueError('FINAL RUN blocked: --confirm-final and --frozen-commit required')
    if args.split=='final' and (args.max_examples or args.smoke_shortest or args.dev_shortest_first
                                or args.lang!='all'):
        raise ValueError('No partial or reordered final run is allowed')
    if args.smoke_shortest and args.dev_shortest_first:
        raise ValueError('Choose either --smoke-shortest or --dev-shortest-first')
    model=os.environ.get('LLM_NAME','swiss-ai/Apertus-v1.5-8B')
    base_url=os.environ.get('LLM_BASE_URL','');key=os.environ.get('LLM_API_KEY','')
    if args.prompt_file:
        prompt=Path(args.prompt_file).read_text(encoding='utf-8')
        prompt_label=str(args.prompt_file)
    else:
        prompt=PAPER_STYLE
        prompt_label='paper-style, NOT verbatim Appendix E'
    signature=digest_settings(model,prompt,args.split,args.max_tokens,base_url,args.frozen_commit)
    # The optional punctuation postprocessing is a distinct experiment.
    # Never mix strict E0 checkpoints with E0-punct-repair checkpoints.
    if args.punctuation_repair:
        signature=hashlib.sha256((signature+'|punctuation-repair-v1').encode()).hexdigest()
    print(f'Method: {prompt_label}; model: {model}; split: {args.split}; signature: {signature[:12]}')
    run_langs=LANGS if args.lang=='all' else (args.lang,)
    for lang in run_langs:
        inputpath=path_in(args.input_dir,args.split,lang)
        rows=list(load_jsonl(inputpath));validate_input_rows(rows)
        cp=Path(args.checkpoint_dir)/f'{args.split}_{lang}.checkpoint.jsonl'
        cp.parent.mkdir(parents=True,exist_ok=True)
        past=read_checkpoint(cp,signature)
        allowed_ids={r['id'] for r in rows}
        if not set(past).issubset(allowed_ids):raise ValueError('Checkpoint includes unknown ID')
        completed=0
        # DEV-only smoke: choose the shortest document-pair to validate the provider cheaply.
        # The selected example remains in its original reference position for checkpointing.
        if args.smoke_shortest:
            selected_rows=[min(rows, key=lambda r:len(r['text_a'].split())+len(r['text_b'].split()))]
        elif args.dev_shortest_first:
            # Development inference order only. Final export still uses original reference order.
            selected_rows=sorted(rows, key=lambda r:(len(r['text_a'].split())+len(r['text_b'].split()), r['id']))
        else:
            selected_rows=rows
        with cp.open('a',encoding='utf-8') as handle:
            for row in selected_rows:
                if row['id'] in past:continue
                if args.max_examples and completed>=args.max_examples:break
                n_a=len(row['text_a'].split());n_b=len(row['text_b'].split())
                host=urlsplit(base_url).hostname or '<unknown>'
                print(f'[INFER START] lang={lang} id={row["id"]} '
                      f'tokens_a={n_a} tokens_b={n_b} total={n_a+n_b} '
                      f'provider={host} max_output_tokens={args.max_tokens}',flush=True)
                started=time.monotonic()
                try:
                    text=chat_completion(base_url,model,encode_prompt(row['text_a'],row['text_b'],prompt),key,
                        timeout=args.timeout,max_tokens=args.max_tokens,retries=args.retries)
                except InferenceError as exc:
                    print(f'[INFER FAILED] {lang}/{row["id"]} after {time.monotonic()-started:.1f}s: {exc}',flush=True)
                    raise
                print(f'[INFER OK] {lang}/{row["id"]} elapsed={time.monotonic()-started:.1f}s '
                      f'output_chars={len(text)}',flush=True)
                repair_log=[]
                try:
                    pred=normalize_prediction(row,text,
                        punctuation_repair=args.punctuation_repair,repair_log=repair_log)
                    valid_prediction(pred,row)
                except (PredictionError,ValueError) as exc:
                    # The response previously vanished on failed parse. Save DEV only;
                    # preserve fail-closed and never write an invalid checkpoint.
                    if args.split=='dev':
                        preserve_invalid_dev_response(lang,row['id'],text,exc,
                            root=BASE/'out'/'diagnostics')
                    raise
                cp_row={'_signature':signature, **pred}
                if args.punctuation_repair:
                    cp_row['_postprocessing']='punctuation-repair-v1'
                    cp_row['_repair_log']=repair_log
                    cp_row['_raw_response_sha256']=hashlib.sha256(text.encode('utf-8')).hexdigest()
                    if any(item['changed'] for item in repair_log):
                        print(f'[PUNCTUATION REPAIR] {row["id"]}: '
                              f'{[item["side"] for item in repair_log if item["changed"]]}',flush=True)
                handle.write(json.dumps(cp_row,ensure_ascii=False)+'\n');handle.flush()
                past[row['id']]=cp_row;completed+=1
                print(f'[RUN {args.split}] {lang} {completed} new / {len(rows)} total | {row["id"]}',flush=True)
        if len(past)==len(rows):
            final=[{k:past[r['id']][k] for k in ('id','text_a','text_b','labels_a','labels_b')} for r in rows]
            outfile=path_out(args.output_dir,lang)
            write_jsonl(outfile,final)
            print(f'[COMPLETE] {lang}: {len(final)} predictions -> {outfile}')
        else:print(f'[PARTIAL] {lang}: {len(past)}/{len(rows)}; final predictions NOT exported')

def do_validate(args):
    for lang in LANGS:
        inp=list(load_jsonl(path_in(args.input_dir,args.split,lang)))
        preds=list(load_jsonl(path_out(args.output_dir,lang)))
        validate_input_rows(inp)
        if len(inp)!=len(preds):raise ValueError(f'{lang}: entries {len(inp)} != predictions {len(preds)}')
        for i,(row,pred) in enumerate(zip(inp,preds)):
            try: valid_prediction(pred,row)
            except ValueError as e:raise ValueError(f'{lang}[{i}] {row["id"]}: {e}') from e
        print(f'[VALID] {lang} {len(inp)} items; token counts, order, IDs, ranges OK')

def do_score_dev(args):
    # No final/test evaluator entry point on purpose.
    root=Path(args.upstream_root).resolve()
    if not (root/'scripts'/'evaluate_predictions_admin.py').exists():
        raise FileNotFoundError('Clone official SwissGov-RSD repository first')
    prefix=(Path(args.output_dir).resolve()/'CLAUSESHIFT_admin_')
    print('[DEV SCORE] Running ORIGINAL evaluator, --split dev (not test)',flush=True)
    subprocess.run([sys.executable,'-m','scripts.evaluate_predictions_admin',str(prefix),'--split','dev'],
                   cwd=root,check=True)

def main(argv=None):
    p=argparse.ArgumentParser(description='CLAUSESHIFT benchmark runner; development/test isolation enforced')
    sp=p.add_subparsers(dest='command',required=True)
    def dirs(s):
        s.add_argument('--input-dir',type=Path,default=BASE/'data'/'inputs')
        s.add_argument('--output-dir',type=Path,default=BASE/'out'/'predictions')
    fetch=sp.add_parser('fetch-dev',help='Download DEV only, never test or full')
    fetch.add_argument('--upstream-sha',default=os.environ.get('UPSTREAM_SHA','master'))
    fetch.add_argument('--input-dir',type=Path,default=BASE/'data'/'inputs')
    fetchf=sp.add_parser('fetch-final',help='FINAL-only fetch of full-order inputs; never develop on these')
    fetchf.add_argument('--upstream-sha',default=os.environ.get('UPSTREAM_SHA','master'))
    fetchf.add_argument('--input-dir',type=Path,default=BASE/'data'/'inputs')
    fetchf.add_argument('--confirm-final',action='store_true');fetchf.add_argument('--frozen-commit')
    prep=sp.add_parser('prepare',help='Create UNLABELED input files from SwissGov reference')
    prep.add_argument('--upstream-root',type=Path,required=True)
    prep.add_argument('--split',choices=['dev','final'],default='dev')
    prep.add_argument('--input-dir',type=Path,default=BASE/'data'/'inputs')
    prep.add_argument('--confirm-final',action='store_true');prep.add_argument('--frozen-commit')
    run=sp.add_parser('run',help='Apertus inference; fail-closed on malformed predictions')
    dirs(run);run.add_argument('--split',choices=['dev','final'],default='dev')
    run.add_argument('--prompt-file',type=Path,help='Optional exact appendix-E prompt template; placeholders required')
    run.add_argument('--checkpoint-dir',type=Path,default=BASE/'out'/'checkpoints')
    run.add_argument('--max-examples',type=int,default=0)
    run.add_argument('--smoke-shortest',action='store_true',help='DEV only: infer shortest document pair per language, save checkpoint')
    run.add_argument('--dev-shortest-first',action='store_true',help='DEV only: process shortest remaining pairs first; does not change final reference order')
    run.add_argument('--lang',choices=['all','de','fr','it'],default='all',help='DEV only: select one language while diagnosing provider limits')
    run.add_argument('--punctuation-repair',action='store_true',help='EXPERIMENTAL: preserve all non-punctuation tokens/scores; restore standalone punctuation as -1')
    run.add_argument('--max-tokens',type=int,default=4096)
    run.add_argument('--timeout',type=float,default=52)
    run.add_argument('--retries',type=int,default=1)
    run.add_argument('--confirm-final',action='store_true');run.add_argument('--frozen-commit')
    val=sp.add_parser('validate');dirs(val);val.add_argument('--split',choices=['dev','final'],default='dev')
    score=sp.add_parser('score-dev');score.add_argument('--upstream-root',type=Path,required=True)
    score.add_argument('--output-dir',type=Path,default=BASE/'out'/'predictions')
    args=p.parse_args(argv)
    try:
        if args.command=='fetch-dev':do_fetch_dev(args)
        elif args.command=='fetch-final':do_fetch_final(args)
        elif args.command=='prepare':do_prepare(args)
        elif args.command=='run':do_run(args)
        elif args.command=='validate':do_validate(args)
        elif args.command=='score-dev':do_score_dev(args)
    except (ValueError,FileNotFoundError,PredictionError,InferenceError,subprocess.CalledProcessError) as e:
        print(f'[FAIL CLOSED] {e}',file=sys.stderr);return 1
    return 0
if __name__=='__main__':sys.exit(main())
