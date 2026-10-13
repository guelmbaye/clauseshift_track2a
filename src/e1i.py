"""E1-I DEV-only indexed Apertus similarity scores, with exact coverage and resumption.

New candidate method, NOT a reproduction of the paper's baseline.
No held-out test files are read or imported. Nothing is padded or invented:
non-punctuation semantic scores only originate from Apertus; punctuation is -1
according to the benchmark convention.
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

from .core import (LANGS, PredictionError, is_punctuation_only, load_jsonl,
                   valid_prediction, validate_input_rows, write_jsonl)
from .e1c import chunk_ranges, context_slice, chunk_hash
from .inference import chat_completion, InferenceError

ROOT = Path(__file__).resolve().parent.parent
VERSION = 'clauseshift-e1i-indexed-selective-v1'


def unique_json(text):
    """Parse complete JSON, rejecting duplicate keys even in nested maps."""
    source = text.strip()
    if source.startswith('```'):
        lines = source.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith('```'):
            raise PredictionError('Unterminated JSON code fence')
        source = '\n'.join(lines[1:-1]).strip()

    def pairs_unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise PredictionError(f'Duplicate JSON key {key!r}')
            out[key] = value
        return out
    try:
        obj = json.loads(source, object_pairs_hook=pairs_unique)
    except json.JSONDecodeError as exc:
        raise PredictionError(f'Invalid JSON: {exc}') from exc
    if not isinstance(obj, dict):
        raise PredictionError('Expected JSON object')
    return obj


def required_indices(core):
    return {i for i, token in enumerate(core) if not is_punctuation_only(token)}


def encode_indexed_prompt(ta, tb, ra, rb, context, requested_a, requested_b):
    a = context_slice(ta, *ra, context)
    b = context_slice(tb, *rb, context)
    payload = {
        'english': {**a, 'core': [{'i': i, 'token': t} for i, t in enumerate(a['core'])]},
        'other_language': {**b, 'core': [{'i': i, 'token': t} for i, t in enumerate(b['core'])]},
        'requested_a': sorted(requested_a), 'requested_b': sorted(requested_b),
    }
    return ("Task: rate semantic SIMILARITY of each requested CORE token against the whole "
            "other-language passage, including additions and omissions. Read the left/right "
            "contexts as reference; do NOT score context tokens. Indices are ZERO-BASED "
            "within each CORE, and may skip punctuation. Return one raw JSON object "
            "with EXACT keys a and b; each must be an object mapping decimal string "
            "indices to numeric scores (0..5). A value of 5 means full equivalence, "
            "4 similar, 3 closely related, 2 somewhat related, 1 weakly related, "
            "0 unmatched/different. Do not output a list. Do not reproduce words. "
            "Do not produce any indices not requested. Do not omit indices. "
            "Example: requested_a=[0,2], requested_b=[0] -> "
            '{"a":{"0":5,"2":0},"b":{"0":5}}. '
            "Return even empty objects if a side has no requested indices. "
            "Payload:\n" + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            + '\nReturn JSON only:')


def parse_partial(text, core_a, core_b, want_a, want_b):
    data = unique_json(text)
    if set(data) != {'a', 'b'}:
        raise PredictionError(f'Unexpected indexed output keys: {sorted(data)}')
    parsed = []
    for key, core, required in (('a', core_a, want_a), ('b', core_b, want_b)):
        obj = data[key]
        if not isinstance(obj, dict):
            raise PredictionError(f'{key} must be an index-to-score object')
        result = {}
        for raw_idx, score in obj.items():
            if not isinstance(raw_idx, str) or not raw_idx.isdecimal() or str(int(raw_idx)) != raw_idx:
                raise PredictionError(f'{key}: invalid canonical index {raw_idx!r}')
            idx = int(raw_idx)
            if idx not in required:
                raise PredictionError(f'{key}: unrequested index {idx} (or punctuation)')
            if idx >= len(core) or is_punctuation_only(core[idx]):
                raise PredictionError(f'{key}: invalid token index {idx}')
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 5:
                raise PredictionError(f'{key}[{idx}]: invalid similarity score {score!r}')
            result[idx] = float(score)
        parsed.append(result)
    if not (parsed[0] or parsed[1]):
        raise PredictionError('No requested scores generated; no progress')
    return parsed[0], parsed[1]


def signature(args, model, endpoint):
    cfg = {'version': VERSION, 'model': model, 'endpoint': endpoint,
           'chunk_tokens': args.chunk_tokens, 'context': args.context,
           'max_tokens': args.max_tokens, 'split': 'dev'}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode('utf-8')).hexdigest()


def read_updates(path, sig, rows_by_id, args):
    """Revalidate each persisted partial score; never accept conflicting scores."""
    state = {}
    if not path.exists():
        return state
    for record in load_jsonl(path):
        if record.get('_signature') != sig:
            raise ValueError('Signature mismatch; choose new indexed checkpoint directory')
        doc_id = record.get('id')
        if doc_id not in rows_by_id:
            raise ValueError(f'Unknown checkpoint ID {doc_id}')
        row = rows_by_id[doc_id]
        ta, tb = row['text_a'].split(), row['text_b'].split()
        windows = chunk_ranges(len(ta), len(tb), args.chunk_tokens)
        idx = record['chunk_index']
        if not isinstance(idx, int) or idx < 0 or idx >= len(windows):
            raise ValueError(f'Invalid chunk index {idx}')
        ra, rb = windows[idx]
        if record.get('ra') != list(ra) or record.get('rb') != list(rb):
            raise ValueError('Stale chunk boundaries')
        if record.get('input_sha256') != chunk_hash(ta, tb, ra, rb, args.context):
            raise ValueError('Input changed since checkpoint')
        core_a, core_b = ta[ra[0]:ra[1]], tb[rb[0]:rb[1]]
        required_a, required_b = required_indices(core_a), required_indices(core_b)
        key = (doc_id, idx)
        partial = state.setdefault(key, {'a': {}, 'b': {}, 'calls': 0})
        for side, required in (('a', required_a), ('b', required_b)):
            saved_map = record.get(side)
            if not isinstance(saved_map, dict):
                raise ValueError('Malformed saved partial scores')
            for raw_idx, score in saved_map.items():
                if not raw_idx.isdecimal() or str(int(raw_idx)) != raw_idx:
                    raise ValueError('Invalid saved index')
                i = int(raw_idx)
                if i not in required:
                    raise ValueError('Saved index is not a semantic token')
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 5:
                    raise ValueError('Invalid saved similarity score')
                if i in partial[side]:
                    raise ValueError('Duplicate saved index: possible corrupt checkpoint')
                partial[side][i] = float(score)
        partial['calls'] += 1
    return state


def completed_document(row, state, args):
    ta, tb = row['text_a'].split(), row['text_b'].split()
    scores_a, scores_b = [], []
    for idx, (ra, rb) in enumerate(chunk_ranges(len(ta), len(tb), args.chunk_tokens)):
        partial = state.get((row['id'], idx), {'a': {}, 'b': {}})
        for side, tokens, destination in (("a", ta[ra[0]:ra[1]], scores_a),
                                          ("b", tb[rb[0]:rb[1]], scores_b)):
            if set(partial[side]) != required_indices(tokens):
                return None
            destination.extend(-1.0 if is_punctuation_only(token) else 1 - partial[side][i] / 5
                               for i, token in enumerate(tokens))
    pred = {'id': row['id'], 'text_a': row['text_a'], 'text_b': row['text_b'],
            'labels_a': scores_a, 'labels_b': scores_b}
    valid_prediction(pred, row)
    return pred


def run(args, infer=chat_completion):
    if args.context < 0 or args.max_calls < 0 or args.max_documents < 0 or args.max_attempts_per_chunk < 1:
        raise ValueError('Invalid nonnegative limits / chunk retry policy')
    model = os.environ.get('LLM_NAME', '')
    endpoint = os.environ.get('LLM_BASE_URL', '')
    token = os.environ.get('LLM_API_KEY', '')
    if 'apertus' not in model.lower() or not endpoint:
        raise ValueError('Set LLM_NAME=Apertus and LLM_BASE_URL')
    sig = signature(args, model, endpoint)
    host = urlsplit(endpoint).hostname or '<unknown>'
    print(f'[E1-I DEV ONLY] model={model} signature={sig[:12]} chunk={args.chunk_tokens} context={args.context}', flush=True)
    n_calls = 0
    langs = LANGS if args.lang == 'all' else (args.lang,)
    for lang in langs:
        allrows = list(load_jsonl(Path(args.input_dir) / f'dev_{lang}.jsonl'))
        validate_input_rows(allrows)
        rows_by_id = {r['id']: r for r in allrows}
        if args.doc_id and args.doc_id not in rows_by_id:
            raise ValueError(f'Document {args.doc_id} not in development {lang}')
        checkpoint = Path(args.checkpoint_dir) / f'dev_{lang}_indexed.checkpoint.jsonl'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        state = read_updates(checkpoint, sig, rows_by_id, args)
        ordered = sorted(allrows, key=lambda r: (len(r['text_a'].split())+len(r['text_b'].split()),r['id'])) if args.shortest_first else allrows
        completed_here = 0
        with checkpoint.open('a', encoding='utf-8') as handle:
            for row in ordered:
                if args.doc_id and row['id'] != args.doc_id:
                    continue
                if args.max_documents and completed_here >= args.max_documents:
                    break
                if completed_document(row, state, args):
                    continue
                ta, tb = row['text_a'].split(), row['text_b'].split()
                windows = chunk_ranges(len(ta), len(tb), args.chunk_tokens)
                for chunk_index, (ra, rb) in enumerate(windows):
                    a_core, b_core = ta[ra[0]:ra[1]], tb[rb[0]:rb[1]]
                    need_a, need_b = required_indices(a_core), required_indices(b_core)
                    key = (row['id'], chunk_index)
                    partial = state.setdefault(key, {'a': {}, 'b': {}, 'calls': 0})
                    while True:
                        missing_a = need_a - set(partial['a'])
                        missing_b = need_b - set(partial['b'])
                        if not (missing_a or missing_b):
                            print(f'[E1-I CHUNK COMPLETE] {lang}/{row["id"]} chunk={chunk_index+1}/{len(windows)}',flush=True)
                            break
                        if partial['calls'] >= args.max_attempts_per_chunk:
                            print(f'[E1-I FAIL CLOSED] {lang}/{row["id"]} chunk={chunk_index+1}: '
                                  f'no complete coverage after {partial["calls"]} calls; '
                                  f'missing_a={sorted(missing_a)} missing_b={sorted(missing_b)}',file=sys.stderr,flush=True)
                            return 1
                        if args.max_calls and n_calls >= args.max_calls:
                            print(f'[E1-I CALL LIMIT] {n_calls} new calls, partial scores saved',flush=True)
                            return 0
                        prompt = encode_indexed_prompt(ta, tb, ra, rb, args.context, missing_a, missing_b)
                        print(f'[E1-I START] {lang}/{row["id"]} chunk={chunk_index+1}/{len(windows)} '
                              f'request_a={len(missing_a)} request_b={len(missing_b)} '
                              f'call={partial["calls"]+1}/{args.max_attempts_per_chunk} provider={host}',flush=True)
                        ts = time.monotonic()
                        text = None
                        try:
                            text = infer(endpoint,model,prompt,token,timeout=args.timeout,
                                         max_tokens=args.max_tokens,retries=args.retries)
                            scores_a,scores_b = parse_partial(text,a_core,b_core,missing_a,missing_b)
                        except (InferenceError,PredictionError,ValueError) as exc:
                            if text is not None:
                                raw_sha = hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]
                                raw_path = Path(args.diagnostics_dir) / (
                                    f'dev_{lang}_{row["id"]}_chunk{chunk_index+1}_{time.time_ns()}_{raw_sha}_raw.txt')
                                raw_path.parent.mkdir(parents=True,exist_ok=True)
                                raw_path.write_text(text,encoding='utf-8')
                                print(f'[E1-I RAW INVALID SAVED] {raw_path}',file=sys.stderr,flush=True)
                            print(f'[E1-I FAIL CLOSED] {lang}/{row["id"]} chunk={chunk_index+1} '
                                  f'elapsed={time.monotonic()-ts:.1f}s: {exc}',file=sys.stderr,flush=True)
                            return 1
                        rec = {'_signature':sig,'id':row['id'],'chunk_index':chunk_index,
                               'ra':list(ra),'rb':list(rb),
                               'input_sha256':chunk_hash(ta,tb,ra,rb,args.context),
                               'a':{str(i):val for i,val in scores_a.items()},
                               'b':{str(i):val for i,val in scores_b.items()},
                               'seconds':round(time.monotonic()-ts,2),'output_chars':len(text)}
                        handle.write(json.dumps(rec,ensure_ascii=False)+'\n')
                        handle.flush()
                        partial['a'].update(scores_a);partial['b'].update(scores_b)
                        partial['calls'] += 1
                        n_calls += 1
                        print(f'[E1-I SAVED] {lang}/{row["id"]} chunk={chunk_index+1} '
                              f'new_a={len(scores_a)} new_b={len(scores_b)} '
                              f'remaining_a={len(missing_a)-len(scores_a)} remaining_b={len(missing_b)-len(scores_b)} '
                              f'elapsed={rec["seconds"]}s',flush=True)
                if completed_document(row, state, args) is None:
                    raise ValueError('Internal indexed coverage failure')
                completed_here += 1
                print(f'[E1-I DOCUMENT COMPLETE] {lang}/{row["id"]}',flush=True)
        all_complete = [completed_document(row,state,args) for row in allrows]
        n_complete = sum(item is not None for item in all_complete)
        print(f'[E1-I STATUS] {lang} {n_complete}/{len(allrows)} DEV complete',flush=True)
        if n_complete == len(allrows):
            output = Path(args.output_dir)/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'
            write_jsonl(output, all_complete)
            print(f'[E1-I FULL DEV EXPORT] {output}',flush=True)
        else:
            print('[E1-I PARTIAL] No official-format predictions exported',flush=True)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description='E1-I: indexed, recoverably sparse, strictly DEV only')
    p.add_argument('--lang', choices=['all',*LANGS], default='de')
    p.add_argument('--doc-id',default='')
    p.add_argument('--input-dir',type=Path,default=ROOT/'data'/'inputs')
    p.add_argument('--checkpoint-dir',type=Path,default=ROOT/'out'/'checkpoints'/'e1i_dev_v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'out'/'predictions_e1i_dev')
    p.add_argument('--diagnostics-dir',type=Path,default=ROOT/'out'/'diagnostics')
    p.add_argument('--chunk-tokens',type=int,default=24)
    p.add_argument('--context',type=int,default=20)
    p.add_argument('--max-tokens',type=int,default=512)
    p.add_argument('--timeout',type=float,default=75)
    p.add_argument('--retries',type=int,default=0)
    p.add_argument('--max-calls',type=int,default=1)
    p.add_argument('--max-documents',type=int,default=1)
    p.add_argument('--max-attempts-per-chunk',type=int,default=3)
    p.add_argument('--shortest-first',action='store_true')
    args=p.parse_args(argv)
    try:
        return run(args)
    except (ValueError,FileNotFoundError) as exc:
        print(f'[E1-I FAIL CLOSED] {exc}',file=sys.stderr,flush=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
