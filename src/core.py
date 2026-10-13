"""Input contracts, baseline prompting and safe label projection for SwissGov-RSD."""
import json
import math
import unicodedata
from pathlib import Path

LANGS = ("de", "fr", "it")

# Paper-style examples are independently authored, NOT the verbatim Appendix E prompt.
PAPER_STYLE = """Annotate two tokenized texts for cross-lingual semantic equivalence.
Return only a JSON object with keys sentence1 and sentence2.
Each key contains every token in original order as [token, similarity_score].
Scores: 5=equivalent, 4=very similar, 3=similar, 2=weakly similar,
1=barely similar, 0=no counterpart, -1=punctuation.
Do not merge, omit, split, reorder, or translate input tokens.
Example 1:
Sentence 1: ["The", "meeting", "starts", "today", "."]
Sentence 2: ["La", "réunion", "commence", "aujourd'hui", "."]
Output: {"sentence1": [["The",5],["meeting",5],["starts",5],["today",5],[".",-1]], "sentence2": [["La",5],["réunion",5],["commence",5],["aujourd'hui",5],[".",-1]]}
Example 2:
Sentence 1: ["Applications", "close", "Monday", "."]
Sentence 2: ["Les", "candidatures", "ferment", "vendredi", "."]
Output: {"sentence1": [["Applications",5],["close",5],["Monday",0],[".",-1]], "sentence2": [["Les",5],["candidatures",5],["ferment",5],["vendredi",0],[".",-1]]}
Example 3:
Sentence 1: ["Payments", "are", "required", "."]
Sentence 2: ["Les", "paiements", "sont", "facultatifs", "."]
Output: {"sentence1": [["Payments",5],["are",5],["required",0],[".",-1]], "sentence2": [["Les",5],["paiements",5],["sont",5],["facultatifs",0],[".",-1]]}
Score every token. Return raw JSON only.
Sentence 1: {{ sentence1 }}
Sentence 2: {{ sentence2 }}
Respond with the JSON object."""

class PredictionError(ValueError): pass

def parse_json_object(text):
    txt=text.strip()
    if txt.startswith('```'):
        txt='\n'.join(txt.splitlines()[1:]); txt=txt.rsplit('```',1)[0].strip()
    try: data=json.loads(txt)
    except json.JSONDecodeError as e: raise PredictionError(f'Non-JSON response: {e}') from e
    if not isinstance(data,dict): raise PredictionError('Expected JSON object')
    return data

def project_token_labels(data, tokens, key):
    """Strictly map paper's similarity labels to divergence. Never silently pad/truncate."""
    seq=data.get(key)
    if not isinstance(seq,list) or len(seq)!=len(tokens):
        raise PredictionError(f'{key}: {len(tokens)} token entries required; got {len(seq) if isinstance(seq,list) else "not a list"}')
    result=[]
    for i,(original,entry) in enumerate(zip(tokens,seq)):
        if not isinstance(entry,(list,tuple)) or len(entry)!=2 or entry[0]!=original:
            raise PredictionError(f'{key}[{i}]: token mismatch / expected [token,score]')
        v=entry[1]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v< -1 or v>5:
            raise PredictionError(f'{key}[{i}]: invalid score {v!r}')
        if v<0 and v!=-1: raise PredictionError('Only -1 permitted below zero')
        result.append(-1.0 if v==-1 else 1.0-float(v)/5.0)
    return result

def encode_prompt(text_a,text_b,template=PAPER_STYLE):
    if '{{ sentence1 }}' not in template or '{{ sentence2 }}' not in template:
        raise ValueError('Template must contain {{ sentence1 }} and {{ sentence2 }}')
    return template.replace('{{ sentence1 }}',json.dumps(text_a.split(),ensure_ascii=False)).replace('{{ sentence2 }}',json.dumps(text_b.split(),ensure_ascii=False))

def is_punctuation_only(token):
    """Conservatively recognize stand-alone Unicode punctuation, not mixed words.

    Example: "," is punctuation, but "l'administration" and "3,5" are not.
    Unknown symbols are *not* silently classified as punctuation.
    """
    return bool(token) and all(unicodedata.category(char).startswith('P') for char in token)


def restore_punctuation_only(expected, generated, side):
    """Restore input punctuation positions without inventing semantic labels.

    The generated sequence must contain the exact same NON-punctuation tokens in
    the exact same order. All model-produced semantic scores are preserved.
    Stand-alone punctuation in the input receives -1; generated punctuation
    is discarded. Refuse malformed/changed/omitted/reordered semantic tokens.
    """
    if not isinstance(generated, list):
        raise PredictionError(f'{side}: expected list of [token, score]')
    for i, entry in enumerate(generated):
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
            raise PredictionError(f'{side}[{i}]: malformed [token, score]')
        v = entry[1]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < -1 or v > 5:
            raise PredictionError(f'{side}[{i}]: invalid score {v!r}')

    exp_semantic = [t for t in expected if not is_punctuation_only(t)]
    gen_semantic = [e for e in generated if not is_punctuation_only(e[0])]
    generated_tokens = [e[0] for e in gen_semantic]
    if generated_tokens != exp_semantic:
        for i, (want, got) in enumerate(zip(exp_semantic, generated_tokens)):
            if want != got:
                raise PredictionError(
                    f'{side}: semantic token drift at non-punctuation position {i}: '
                    f'expected {want!r}, got {got!r}. Refusing repair.')
        raise PredictionError(
            f'{side}: semantic token count changed: expected {len(exp_semantic)}, '
            f'got {len(generated_tokens)}. Refusing repair.')
    if any(e[1] == -1 for e in gen_semantic):
        raise PredictionError(f'{side}: semantic token was given punctuation score -1; refusing repair')

    it = iter(gen_semantic)
    restored = [[token, -1] if is_punctuation_only(token) else next(it)
                for token in expected]
    changed = restored != generated
    return restored, {
        'side': side,
        'changed': changed,
        'source_punctuation': sum(is_punctuation_only(t) for t in expected),
        'generated_punctuation': len(generated) - len(gen_semantic),
        'semantic_tokens_preserved': len(exp_semantic),
    }


def remove_one_extraneous_rparen(text):
    """One conservative syntax fix: drop a stray ')' in `]), [` outside strings.

    Refuse any other syntax error, multiple candidates, non-JSON output,
    or modification inside quoted tokens. Semantic validation is separate.
    """
    try:
        parse_json_object(text)
        return text, {'changed': False, 'kind': 'none'}
    except PredictionError as exc:
        decode_error = exc.__cause__
        if not isinstance(decode_error, json.JSONDecodeError):
            raise

    # Character-by-character tracking: do not match punctuation inside JSON strings.
    inside = False
    escaped = False
    candidates = []
    for i, char in enumerate(text):
        if inside:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                inside = False
            continue
        if char == '"':
            inside = True
            continue
        if char != ')' or i == 0 or text[i - 1] != ']':
            continue
        # The observed error was: ["sector", 5]), [")", -1]
        # Removing ONLY ')' gives: ["sector", 5], [")", -1]
        j = i + 1
        if j >= len(text) or text[j] != ',':
            continue
        j += 1
        while j < len(text) and text[j].isspace():
            j += 1
        if j < len(text) and text[j] == '[':
            candidates.append(i)

    if len(candidates) != 1:
        raise PredictionError(
            f'Syntax repair refused: expected one stray bracket-adjacent ")"; found {len(candidates)}') from decode_error
    pos = candidates[0]
    if abs(pos - decode_error.pos) > 2:
        raise PredictionError('Syntax repair refused: candidate is not at parser failure') from decode_error
    amended = text[:pos] + text[pos + 1:]
    # Validity is checked across the ENTIRE object, not only near the error.
    parse_json_object(amended)
    return amended, {'changed': True, 'kind': 'removed-stray-rparen',
                     'raw_character_offset': pos}


def normalize_prediction(input_row, response, punctuation_repair=False, repair_log=None,
                         syntax_repair=False, syntax_log=None):
    if syntax_repair:
        response, info = remove_one_extraneous_rparen(response)
        if syntax_log is not None:
            syntax_log.append(info)
    data=parse_json_object(response)
    if punctuation_repair:
        for key, text in (('sentence1', input_row['text_a']), ('sentence2', input_row['text_b'])):
            fixed, info = restore_punctuation_only(text.split(), data.get(key), key)
            data[key] = fixed
            if repair_log is not None: repair_log.append(info)
    return {'id':input_row['id'],'text_a':input_row['text_a'],'text_b':input_row['text_b'],
            'labels_a':project_token_labels(data,input_row['text_a'].split(),'sentence1'),
            'labels_b':project_token_labels(data,input_row['text_b'].split(),'sentence2')}

def valid_prediction(row,original):
    if not isinstance(row,dict): raise ValueError('Prediction must be object')
    for k in ('id','text_a','text_b'):
        if row.get(k)!=original[k]: raise ValueError(f'{k}: mismatch')
    for side,label in (('text_a','labels_a'),('text_b','labels_b')):
        vals=row.get(label)
        if not isinstance(vals,list) or len(vals)!=len(original[side].split()):
            raise ValueError(f'{label}: wrong token count')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v< -1 or v>1 for v in vals):
            raise ValueError(f'{label}: invalid divergence score')

def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{n}: {e}') from e

def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')

def validate_input_rows(rows):
    seen=set()
    for r in rows:
        if not all(isinstance(r.get(k),str) and r[k] for k in ('id','text_a','text_b')): raise ValueError('Missing id/text_a/text_b')
        if r['id'] in seen:raise ValueError(f'Duplicate id: {r["id"]}')
        seen.add(r['id'])
    if not rows:raise ValueError('No rows found')
