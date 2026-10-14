"""Frozen E0-PJQ submission driver.

No test/final inputs are touched by `freeze` or `preflight`.
Only `run` can retrieve full-order inputs, and only after manifest verification.
Do not interpret completion as quality validation on held-out data.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from . import cli
from .core import LANGS

BASE = Path(__file__).resolve().parent.parent
MANIFEST = BASE / 'docs' / 'submission_frozen.json'
METHOD = 'E0-PJQ'
BATCH_METHOD = 'E2-BATCH'
SINGLE_METHOD = 'E2-SINGLE'  # Validated indexed per-unit format (no multi-unit JSON)
BATCH_SCHEMA_KEYS = {'alignment','prompt','embedding_model','embedding_revision',
 'chunk_tokens','max_unit_tokens','similarity_threshold','merge_penalty',
 'coverage_weight','anchor_bonus','context','max_scores_per_call',
 'max_units_per_batch','workers','launch_interval','max_attempts_per_unit'}



def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_digest(root=BASE):
    root = Path(root)
    sources = sorted((root / 'src').glob('*.py'))
    sources += [root / 'Makefile', root / 'Dockerfile', root / 'requirements.txt']
    h = hashlib.sha256()
    for p in sources:
        if not p.is_file():
            raise ValueError(f'Missing submission code: {p}')
        rel = p.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(4, 'big') + rel)
        h.update(bytes.fromhex(sha_file(p)))
    return h.hexdigest()


def freeze(manifest, upstream_sha, frozen_commit, model, max_tokens=4096, timeout=75, retries=0,
           *, method=METHOD, embedding_revision='', batch_cap=96,
           units_per_batch=3, workers=1, launch_interval=0):
    if not re.fullmatch(r'[a-fA-F0-9]{40}', upstream_sha or ''):
        raise ValueError('Pin the exact 40-char upstream Git SHA; no master/main')
    if not re.fullmatch(r'[a-fA-F0-9]{7,40}', frozen_commit or ''):
        raise ValueError('Specify real frozen Git revision with --frozen-commit')
    if 'apertus' not in (model or '').lower():
        raise ValueError('An Apertus model is mandatory')
    if max_tokens < 512 or timeout <= 0 or retries < 0:
        raise ValueError('Invalid runtime configuration')
    prompt = BASE / 'prompts' / 'paper_exact.txt'
    if not prompt.is_file() or '{{ sentence1 }}' not in prompt.read_text() or '{{ sentence2 }}' not in prompt.read_text():
        raise ValueError('Exact baseline prompt missing placeholders')
    path = Path(manifest)
    if path.exists():
        raise ValueError('Frozen manifest already exists: refuse overwrite')
    if method not in (METHOD,BATCH_METHOD,SINGLE_METHOD):
        raise ValueError('Unknown submission method')
    if method in (BATCH_METHOD,SINGLE_METHOD):
        if not re.fullmatch(r'[a-f0-9]{40}',embedding_revision.lower()):
            raise ValueError('Pin a full 40-character HF model revision for batching')
        if not 2<=batch_cap<=128 or not 1<=units_per_batch<=16 or not 1<=workers<=8 or launch_interval<0:
            raise ValueError('Invalid batch/final concurrency limits')
    payload = {
        'schema': 1,
        'method': method,
        'model': model,
        'upstream_sha': upstream_sha.lower(),
        'frozen_commit': frozen_commit.lower(),
        'code_sha256': code_digest(),
        'prompt_file': 'prompts/paper_exact.txt',
        'prompt_sha256': sha_file(prompt),
        'max_tokens': max_tokens,
        'timeout': timeout,
        'retries': retries,
        'punctuation_repair': True,
        'syntax_repair': True,
        'boundary_quote_repair': True,
    }
    if method in (BATCH_METHOD,SINGLE_METHOD):
        payload.update(punctuation_repair=False,syntax_repair=False,boundary_quote_repair=False)
        payload['schema']=3 if method==SINGLE_METHOD else 2
        payload['e2']={
            'alignment':'semantic_coverage','prompt':'fewshot',
            'embedding_model':'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'embedding_revision':embedding_revision.lower(),
            'chunk_tokens':48,'max_unit_tokens':48,'similarity_threshold':0.46,
            'merge_penalty':0.06,'coverage_weight':0.50,'anchor_bonus':0.35,
            'context':20,'max_scores_per_call':batch_cap,
            'max_units_per_batch':(1 if method==SINGLE_METHOD else units_per_batch),'workers':(1 if method==SINGLE_METHOD else workers),
            'launch_interval':float(launch_interval),'max_attempts_per_unit':8,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'[FREEZE CREATED] {path}: method={method}, code={payload["code_sha256"][:12]}')
    print('[FREEZE] Commit this manifest to the final repository; do not change code or prompt afterwards')
    return payload


def preflight(manifest=MANIFEST, *, require_secrets=False):
    path = Path(manifest)
    if not path.is_file():
        raise ValueError(f'Missing frozen manifest {path}; refusing final/test access')
    data = json.loads(path.read_text(encoding='utf-8'))
    expected = {'schema','method','model','upstream_sha','frozen_commit','code_sha256',
                'prompt_file','prompt_sha256','max_tokens','timeout','retries',
                'punctuation_repair','syntax_repair','boundary_quote_repair'}
    batch=(data.get('schema')==2 and data.get('method')==BATCH_METHOD)
    single=(data.get('schema')==3 and data.get('method')==SINGLE_METHOD)
    if batch or single:
        if set(data) != expected | {'e2'} or not isinstance(data.get('e2'),dict):
            raise ValueError('Unexpected E2-manifest schema')
        e2=data['e2']
        if set(e2)!=BATCH_SCHEMA_KEYS or e2['alignment']!='semantic_coverage' or e2['prompt']!='fewshot':
            raise ValueError('Frozen batched method cannot be changed')
        if (not re.fullmatch(r'[a-f0-9]{40}',e2['embedding_revision']) or
            e2['embedding_model']!='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' or
            not 2<=e2['max_scores_per_call']<=128 or not 1<=e2['max_units_per_batch']<=16 or
            not 1<=e2['workers']<=8 or e2['launch_interval']<0 or
            e2['max_attempts_per_unit']<1):
            raise ValueError('Invalid frozen E2 settings')
        if single and (e2['max_units_per_batch'] != 1 or e2['workers'] != 1):
            raise ValueError('Single-unit method cannot use batching or concurrency')
    elif set(data) != expected or data.get('schema') != 1 or data.get('method') != METHOD:
        raise ValueError('Unexpected frozen-manifest schema or method')
    if not re.fullmatch(r'[a-f0-9]{40}', data['upstream_sha']):
        raise ValueError('Upstream Git SHA must be pinned')
    if not re.fullmatch(r'[a-f0-9]{7,40}', data['frozen_commit']):
        raise ValueError('Frozen Git SHA missing')
    if (data['prompt_file'] != 'prompts/paper_exact.txt' or
            sha_file(BASE / data['prompt_file']) != data['prompt_sha256']):
        raise ValueError('Frozen prompt changed')
    if data['code_sha256'] != code_digest():
        raise ValueError('Submission code changed after freeze')
    flags=(data['punctuation_repair'],data['syntax_repair'],data['boundary_quote_repair'])
    if flags != ((False,False,False) if (batch or single) else (True,True,True)):
        raise ValueError('Frozen repair configuration differs from selected method')
    if type(data['max_tokens']) is not int or data['max_tokens'] < 512:
        raise ValueError('Invalid max_tokens')
    if type(data['timeout']) not in (int, float) or data['timeout'] <= 0:
        raise ValueError('Invalid timeout')
    if type(data['retries']) is not int or data['retries'] < 0:
        raise ValueError('Invalid retries')
    if 'apertus' not in data['model'].lower():
        raise ValueError('Primary inference model is not Apertus')
    if require_secrets:
        if os.environ.get('LLM_NAME') != data['model']:
            raise ValueError('LLM_NAME is different from frozen model')
        if not os.environ.get('LLM_BASE_URL') or not os.environ.get('LLM_API_KEY'):
            raise ValueError('LLM_BASE_URL and LLM_API_KEY are required')
    print(f'[PREFLIGHT OK] {data["method"]} code={data["code_sha256"][:12]} upstream={data["upstream_sha"][:12]}')
    return data


def run_final(manifest=MANIFEST):
    data = preflight(manifest, require_secrets=True)
    input_dir = BASE / 'data' / 'inputs'
    output_dir = BASE / 'out' / 'predictions'
    cpdir = BASE / 'out' / 'checkpoints' / 'final_e0pjq'
    # Do not accept stale predictions from another run merely because IDs match.
    for lang in LANGS:
        p = cli.path_out(output_dir, lang)
        if p.exists():
            p.unlink()
    cli.do_fetch_final(SimpleNamespace(upstream_sha=data['upstream_sha'], input_dir=input_dir,
                                       confirm_final=True, frozen_commit=data['frozen_commit']))
    if data['method']==SINGLE_METHOD:
        from . import e2
        settings=dict(data['e2'])
        settings.pop('max_units_per_batch')
        settings.pop('workers')
        settings.pop('launch_interval')
        settings.update(lang='all',doc_id='',input_dir=input_dir,output_dir=output_dir,
                        checkpoint_dir=BASE/'out'/'checkpoints'/'final_e2single_v1143',
                        plan_dir=BASE/'out'/'plans'/'final_e2single_v1143',
                        diagnostics_dir=BASE/'out'/'diagnostics',
                        timeout=data['timeout'],max_tokens=data['max_tokens'],
                        retries=data['retries'],max_calls=0,max_documents=0,
                        shortest_first=False,plan_only=False,
                        frozen_commit=data['frozen_commit'],_frozen_authorized=True)
        status=e2.run(SimpleNamespace(**settings),split='final')
        if status:
            raise ValueError('Single-unit final inference failed; no successful final submission')
    elif data['method']==BATCH_METHOD:
        from . import e2_batch
        settings=dict(data['e2'])
        settings.update(lang='all',doc_id='',input_dir=input_dir,output_dir=output_dir,
                        checkpoint_dir=BASE/'out'/'checkpoints'/'final_e2batch_v114',
                        plan_dir=BASE/'out'/'plans'/'final_e2batch_v114',
                        timeout=data['timeout'],max_tokens=data['max_tokens'],
                        retries=data['retries'],max_calls=0,max_documents=0,
                        shortest_first=False,plan_only=False,
                        frozen_commit=data['frozen_commit'],_frozen_authorized=True)
        status=e2_batch.run_dataset(SimpleNamespace(**settings),split='final')
        if status:
            raise ValueError('Batched final inference failed; no successful final submission')
    else:
        cli.do_run(SimpleNamespace(
            split='final', input_dir=input_dir, output_dir=output_dir,
            checkpoint_dir=cpdir, prompt_file=BASE / data['prompt_file'],
            max_examples=0, smoke_shortest=False, dev_shortest_first=False, lang='all',
            punctuation_repair=True, syntax_repair=True, boundary_quote_repair=True,
            max_tokens=data['max_tokens'], timeout=data['timeout'], retries=data['retries'],
            confirm_final=True, frozen_commit=data['frozen_commit']))
    # `do_run` exports only complete sets; this is the final fail-closed gate.
    cli.do_validate(SimpleNamespace(split='final',input_dir=input_dir,output_dir=output_dir))
    for lang in LANGS:
        p = cli.path_out(output_dir, lang)
        print(f'[FINAL COMPLETE] {lang}: {p} sha256={sha_file(p)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Frozen E0-PJQ, E2-SINGLE or E2-BATCH submission (final data only after freeze)')
    cmd = parser.add_subparsers(dest='action',required=True)
    f = cmd.add_parser('freeze')
    f.add_argument('--upstream-sha',required=True)
    f.add_argument('--frozen-commit',required=True)
    f.add_argument('--model',default=os.environ.get('LLM_NAME',''))
    f.add_argument('--max-tokens',type=int,default=4096)
    f.add_argument('--timeout',type=float,default=75)
    f.add_argument('--retries',type=int,default=0)
    f.add_argument('--method',choices=[METHOD,BATCH_METHOD,SINGLE_METHOD],default=METHOD)
    f.add_argument('--embedding-revision',default=os.environ.get('EMB_SHA',''))
    f.add_argument('--batch-cap',type=int,default=96)
    f.add_argument('--units-per-batch',type=int,default=3)
    f.add_argument('--workers',type=int,default=1)
    f.add_argument('--launch-interval',type=float,default=0)
    for name in ('freeze','preflight','run'):
        # `freeze` receives these options too; all default to same path.
        (f if name=='freeze' else cmd.add_parser(name)).add_argument('--manifest',type=Path,default=MANIFEST)
    args=parser.parse_args(argv)
    try:
        if args.action=='freeze':freeze(args.manifest,args.upstream_sha,args.frozen_commit,args.model,args.max_tokens,args.timeout,args.retries,
                                          method=args.method,embedding_revision=args.embedding_revision,
                                          batch_cap=args.batch_cap,units_per_batch=args.units_per_batch,
                                          workers=args.workers,launch_interval=args.launch_interval)
        elif args.action=='preflight':preflight(args.manifest)
        elif args.action=='run':run_final(args.manifest)
    except (ValueError,FileNotFoundError,cli.InferenceError,cli.PredictionError) as exc:
        print(f'[FINAL FAIL CLOSED] {exc}',file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    sys.exit(main())
