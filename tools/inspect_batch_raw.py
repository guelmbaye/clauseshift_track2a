"""Read-only diagnostics for malformed DEV batch model responses.

Never emits source text or scores; shows structural characteristics only.
Does not load benchmark annotations, API credentials or checkpoints.
"""
import argparse
import json
from pathlib import Path


def analyze_raw(raw):
    text = raw.strip()
    result = {'length': len(raw), 'starts_with_fence': text.startswith('```')}
    try:
        full = json.loads(text)
    except json.JSONDecodeError as exc:
        result['json_error'] = exc.msg
        result['error_offset'] = exc.pos
        result['error_at_end'] = exc.pos >= len(text) - 30
    else:
        result['full_json_valid'] = True
        result['root_type'] = type(full).__name__
        if isinstance(full, dict):
            result['root_keys'] = sorted(map(str,full))
            if isinstance(full.get('units'), dict):
                result['unit_keys'] = sorted(map(str, full['units']))
        return result
    try:
        obj,end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        result['valid_prefix'] = False
        return result
    extra = text[end:].strip()
    result['valid_prefix'] = True
    result['prefix_type'] = type(obj).__name__
    result['prefix_length'] = end
    if isinstance(obj,dict):
        result['prefix_keys'] = sorted(map(str,obj))
        if isinstance(obj.get('units'),dict):
            result['prefix_unit_keys'] = sorted(map(str,obj['units']))
    result['extra_length'] = len(extra)
    if not extra:
        result['extra_type'] = 'whitespace_only'
    elif extra.startswith('{'):
        try:
            second, second_end = json.JSONDecoder().raw_decode(extra)
        except json.JSONDecodeError:
            result['extra_type'] = 'incomplete_or_malformed_object'
        else:
            result['extra_type'] = 'second_json_value'
            result['second_type'] = type(second).__name__
            result['second_length'] = second_end
            if isinstance(second,dict):
                result['second_keys'] = sorted(map(str,second))
                if isinstance(second.get('units'),dict):
                    result['second_unit_keys'] = sorted(map(str, second['units']))
            result['after_second_length'] = len(extra[second_end:].strip())
    elif extra.startswith('```'):
        result['extra_type'] = 'markdown_fence'
    else:
        result['extra_type'] = 'other_text_or_value'
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description='Inspect malformed DEV batch output structure offline')
    p.add_argument('--raw-file',type=Path,required=True)
    a=p.parse_args(argv)
    text=a.raw_file.read_text(encoding='utf-8')
    print('[DEV RAW STRUCTURE] '+json.dumps(analyze_raw(text),sort_keys=True))
    print('[READ ONLY] No score exported, no checkpoint changed, no API request')


if __name__=='__main__':main()
