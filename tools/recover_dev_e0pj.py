#!/usr/bin/env python3
"""DEV-only replay of one saved malformed response + validated E0-P checkpoint migration.

Do not use test/final here; do not call an API or change the original E0-P files.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.cli import digest_settings, read_checkpoint
from src.core import load_jsonl, normalize_prediction, valid_prediction, write_jsonl, LANGS


def sigs(model, prompt, max_tokens, base_url, boundary_quote_repair=False):
    seed = digest_settings(model, prompt, 'dev', max_tokens, base_url, None)
    ep = hashlib.sha256((seed+'|punctuation-repair-v1').encode()).hexdigest()
    epj = hashlib.sha256((ep+'|syntax-repair-v1').encode()).hexdigest()
    epjq = hashlib.sha256((epj+'|boundary-quote-repair-v1').encode()).hexdigest()
    return ep, epj if not boundary_quote_repair else epjq


def recover(args):
    if args.lang not in LANGS:
        raise ValueError('Only de/fr/it allowed')
    raw_file = args.raw_file.resolve(strict=True)
    if f'dev_{args.lang}_{args.doc_id}_' not in raw_file.name or not raw_file.name.endswith('_raw.txt'):
        raise ValueError('Raw filename does not identify requested DEV row')
    original = args.source_checkpoints.resolve(strict=True)
    target = args.destination_checkpoints.resolve()
    if target == original:
        raise ValueError('Must use a NEW checkpoint directory for E0-PJ')
    if target.exists() and any(target.iterdir()):
        raise ValueError('Refusing nonempty E0-PJ destination: do not overwrite checkpoints')

    model = os.environ.get('LLM_NAME')
    base_url = os.environ.get('LLM_BASE_URL')
    if not model or not base_url or 'apertus' not in model.lower():
        raise ValueError('Set the exact LLM_NAME and LLM_BASE_URL used in E0-P')
    prompt = args.prompt_file.read_text(encoding='utf-8')
    old_sig, new_sig = sigs(model, prompt, args.max_tokens, base_url,
                            boundary_quote_repair=getattr(args, "boundary_quote_repair", False))
    staged = {}
    originals = {}
    for lang in LANGS:
        inp = {r['id']: r for r in load_jsonl(ROOT/'data'/'inputs'/f'dev_{lang}.jsonl')}
        originals[lang] = inp
        source = original/f'dev_{lang}.checkpoint.jsonl'
        old_rows = read_checkpoint(source, old_sig)
        if not set(old_rows).issubset(inp):
            raise ValueError(f'{lang}: E0-P checkpoint has unknown IDs')
        converted = []
        for id_, row in old_rows.items():
            valid_prediction(row, inp[id_])
            cp_row = dict(row)
            cp_row['_signature'] = new_sig
            cp_row['_migrated_from'] = 'e0-p-punctuation-repair-v1'
            cp_row['_postprocessing'] = 'punctuation-repair-v1+syntax-repair-v1'+('+boundary-quote-repair-v1' if getattr(args, 'boundary_quote_repair', False) else '')
            cp_row['_syntax_repair_log'] = [{'kind': 'none', 'changed': False}]
            converted.append(cp_row)
        staged[lang] = converted

    inp = originals[args.lang]
    if args.doc_id not in inp:
        raise ValueError(f'Unknown DEV ID: {args.doc_id}')
    if any(x['id'] == args.doc_id for x in staged[args.lang]):
        raise ValueError(f'{args.doc_id} already checkpointed in source')
    raw = raw_file.read_text(encoding='utf-8')
    repair_log, syntax_log = [], []
    pred = normalize_prediction(inp[args.doc_id], raw,
        punctuation_repair=True, repair_log=repair_log,
        syntax_repair=True, syntax_log=syntax_log,
        boundary_quote_repair=getattr(args, "boundary_quote_repair", False))
    valid_prediction(pred, inp[args.doc_id])
    if not any(log.get('changed') for log in syntax_log):
        raise ValueError('This recovery expected a syntax repair, but none was made')
    sha = hashlib.sha256(raw.encode()).hexdigest()
    if f'_{sha[:10]}_raw.txt' not in raw_file.name:
        raise ValueError('Raw filename hash does not match file content')
    cp_new = {'_signature': new_sig, **pred,
              '_postprocessing': 'punctuation-repair-v1+syntax-repair-v1'+('+boundary-quote-repair-v1' if getattr(args, 'boundary_quote_repair', False) else ''),
              '_repair_log': repair_log, '_syntax_repair_log': syntax_log,
              '_raw_response_sha256': sha}
    staged[args.lang].append(cp_new)
    print(f'[OFFLINE VERIFIED] {args.lang}/{args.doc_id}: token counts and scores valid; '
          f'syntax={syntax_log}; punctuation={[x for x in repair_log if x["changed"]]}')
    print(f'[CHECKPOINT SIGNATURES] E0-P {old_sig[:12]} -> E0-PJ{"Q" if getattr(args, "boundary_quote_repair", False) else ""} {new_sig[:12]}')
    print(f'[READY] { {lang:len(staged[lang]) for lang in LANGS} } (DEV only; no API calls)')
    if not args.write:
        print('[DRY RUN] Use --write to create new E0-PJ checkpoint files only after reviewing this output')
        return

    target.mkdir(parents=True, exist_ok=True)
    for lang, rows in staged.items():
        outfile = target/f'dev_{lang}.checkpoint.jsonl'
        if outfile.exists():
            raise ValueError(f'Refusing to overwrite {outfile}')
        write_jsonl(outfile, rows)
        os.chmod(outfile, 0o600)
        print(f'[E0-PJ SAVED] {lang}: {len(rows)} -> {outfile}')
    print('[SAFE] Original E0-P checkpoint files remain unchanged. No test data touched.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Offline DEV recovery, no API calls, separate E0-PJ checkpoint.')
    p.add_argument('--source-checkpoints',type=Path,required=True)
    p.add_argument('--destination-checkpoints',type=Path,required=True)
    p.add_argument('--raw-file',type=Path,required=True)
    p.add_argument('--lang',choices=LANGS,required=True)
    p.add_argument('--doc-id',required=True)
    p.add_argument('--prompt-file',type=Path,required=True)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--boundary-quote-repair',action='store_true',help='Experimental: restore one missing boundary double quote; E0-PJQ separate checkpoint')
    p.add_argument('--write',action='store_true',help='After successful dry-run, write separate DEV E0-PJ/PJQ checkpoints')
    try: recover(p.parse_args())
    except Exception as exc:
        print(f'[FAIL CLOSED] {exc}',file=sys.stderr)
        sys.exit(1)
