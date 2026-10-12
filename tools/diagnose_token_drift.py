#!/usr/bin/env python3
"""CLAUSESHIFT: inspect ONE dev response without altering baseline predictions.

Only accesses prepared dev inputs. A fresh probe incurs exactly one API request.
Run from CLAUSESHIFT project root: python3 tools/diagnose_token_drift.py --lang de --id admin_de_105
"""
import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.core import load_jsonl, encode_prompt, parse_json_object, normalize_prediction, PredictionError
from src.inference import chat_completion


def compare(expected, entries, side):
    if not isinstance(entries, list):
        return {'side': side, 'expected_count': len(expected), 'actual_count': None,
                'error': 'Missing or non-list response field', 'differences': []}
    tokens = [entry[0] if isinstance(entry, list) and len(entry) >= 1 else '<MALFORMED>'
              for entry in entries]
    matcher = difflib.SequenceMatcher(a=expected, b=tokens, autojunk=False)
    differences = []
    for op, ai, aj, bi, bj in matcher.get_opcodes():
        if op == 'equal':
            continue
        differences.append({
            'type': op,
            'expected_positions': [ai, aj],
            'generated_positions': [bi, bj],
            'expected': expected[ai:aj],
            'generated': tokens[bi:bj],
            'context_expected': expected[max(0, ai-3):min(len(expected), aj+3)],
            'context_generated': tokens[max(0, bi-3):min(len(tokens), bj+3)]
        })
    return {'side':side, 'expected_count':len(expected), 'actual_count':len(entries),
            'differences':differences[:8], 'total_difference_blocks':len(differences)}


def analyze_row(row, raw):
    try:
        data = parse_json_object(raw)
    except PredictionError as e:
        return {'valid':False, 'error':str(e), 'raw_is_json':False}
    sides = {
        'sentence1': compare(row['text_a'].split(), data.get('sentence1'), 'sentence1'),
        'sentence2': compare(row['text_b'].split(), data.get('sentence2'), 'sentence2')
    }
    try:
        normalize_prediction(row, raw)
        valid, error = True, None
    except PredictionError as e:
        valid, error = False, str(e)
    return {'valid':valid, 'error':error, 'raw_is_json':True, 'sides':sides}


def run(args):
    if args.id and args.id.lower().startswith('test_'):
        raise SystemExit('Safety guard: test samples are forbidden during development')
    input_path = Path(args.input_dir)/f'dev_{args.lang}.jsonl'
    if not input_path.is_file():
        raise SystemExit(f'Missing DEV input: {input_path}; first run python3 -m src.cli fetch-dev')
    rows = list(load_jsonl(input_path))
    if args.smoke_shortest:
        if not rows: raise SystemExit(f'Empty DEV input: {input_path}')
        row = min(rows, key=lambda r:len(r['text_a'].split())+len(r['text_b'].split()))
    else:
        row = next((r for r in rows if r['id']==args.id), None)
    if row is None:
        raise SystemExit(f'No matching DEV id: {args.id}')
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    filename_id=re.sub(r'[^A-Za-z0-9_-]', '_', row['id'])
    if args.raw_file:
        raw = Path(args.raw_file).read_text(encoding='utf-8')
        source='existing raw response (no new API call)'
    else:
        if not args.confirm_api_call:
            raise SystemExit('A new paid API request is required. Re-run with --confirm-api-call.')
        model=os.environ.get('LLM_NAME','')
        base_url=os.environ.get('LLM_BASE_URL','')
        api_key=os.environ.get('LLM_API_KEY','')
        if not model or not base_url or not api_key:
            raise SystemExit('Set LLM_NAME, LLM_BASE_URL and LLM_API_KEY first')
        prompt_template=Path(args.prompt_file).read_text(encoding='utf-8')
        prompt=encode_prompt(row['text_a'],row['text_b'],prompt_template)
        raw=chat_completion(base_url,model,prompt,api_key,timeout=args.timeout,
                            max_tokens=args.max_tokens,retries=0)
        stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        raw_path=output_dir/f'{args.lang}_{filename_id}_{stamp}_raw.txt'
        raw_path.write_text(raw,encoding='utf-8')
        source=f'one fresh Apertus request; raw saved at {raw_path}'
    report=analyze_row(row,raw)
    report.update({'lang':args.lang,'id':row['id'], 'source':source,
                   'response_sha256':hashlib.sha256(raw.encode()).hexdigest()})
    report_path=output_dir/f'{args.lang}_{filename_id}_diagnosis.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[DEV ONLY] {row["id"]} ({args.lang}) — {source}')
    if 'sides' in report:
        for side,entry in report['sides'].items():
            print(f'[{side}] expected={entry["expected_count"]}, got={entry["actual_count"]}, diff_blocks={entry.get("total_difference_blocks",0)}')
            for diff in entry['differences']:
                print(f'  {diff["type"]} at expected[{diff["expected_positions"][0]}:{diff["expected_positions"][1]}] '
                      f'vs generated[{diff["generated_positions"][0]}:{diff["generated_positions"][1]}]')
                print('    expected:',json.dumps(diff['context_expected'],ensure_ascii=False))
                print('    received:',json.dumps(diff['context_generated'],ensure_ascii=False))
    print(f'[VALID] {report["valid"]}; {report["error"] or "all tokens aligned"}')
    print(f'[DIAGNOSTIC] {report_path}')
    print('[SAFE] No official predictions or checkpoints were modified.')
    return 0 if report['valid'] else 2


def main():
    p=argparse.ArgumentParser(description='Diagnose one malformed Apertus DEV response')
    p.add_argument('--lang',choices=('de','fr','it'),required=True)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--id')
    selection.add_argument('--smoke-shortest',action='store_true',help='Select the shortest DEV pair, just as src.cli run --smoke-shortest does')
    p.add_argument('--input-dir',default='data/inputs')
    p.add_argument('--output-dir',default='out/diagnostics')
    p.add_argument('--prompt-file',default='prompts/paper_exact.txt')
    p.add_argument('--raw-file',help='Analyze a previously captured response without an API call')
    p.add_argument('--confirm-api-call',action='store_true')
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--timeout',type=float,default=75)
    args=p.parse_args()
    return run(args)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        print(f'[DIAGNOSTIC ERROR] {e}',file=sys.stderr)
        sys.exit(1)
