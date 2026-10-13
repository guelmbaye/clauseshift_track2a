#!/usr/bin/env python3
"""Analyze an existing malformed DEV response; NO API, NO data/test access."""
import argparse
import json
import re
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='Inspect saved malformed DEV JSON offline')
    parser.add_argument('raw_file',type=Path)
    parser.add_argument('--radius',type=int,default=140)
    args=parser.parse_args()
    if args.radius < 20 or args.radius > 1000: parser.error('--radius must be between 20 and 1000')
    if not re.fullmatch(r'(?:dev_)?(?:de|fr|it)_admin_(?:de|fr|it)_[A-Za-z0-9_-]+_raw\.txt', args.raw_file.name):
        parser.error('Only named DEV raw diagnostics are accepted')
    if not args.raw_file.is_file(): parser.error('No such raw file')
    txt=args.raw_file.read_text(encoding='utf-8')
    print(f'[OFFLINE DEV] {args.raw_file} chars={len(txt)}')
    try:
        content=json.loads(txt)
    except json.JSONDecodeError as e:
        print(f'[JSON ERROR] {e.msg}; line={e.lineno}; column={e.colno}; char={e.pos}')
        left=max(0,e.pos-args.radius);right=min(len(txt),e.pos+args.radius)
        excerpt=txt[left:right]
        print(f'[EXCERPT] {excerpt!r}')
        print('[RAW END]',repr(txt[-180:]))
        print('[NOTICE] Do not infer missing semantic scores or patch tokens from context.')
        return 2
    if not isinstance(content,dict):
        print('[STRUCTURE] JSON is valid but root is not an object');return 2
    print('[JSON] valid; keys:', sorted(content))
    for key in ('sentence1','sentence2'):
        seq=content.get(key)
        print(f'  {key}: entries={len(seq) if isinstance(seq,list) else "invalid"}')
    return 0
if __name__=='__main__':raise SystemExit(main())
