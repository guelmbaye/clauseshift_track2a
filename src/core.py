"""Input contracts, baseline prompting and safe label projection for SwissGov-RSD."""
import json
import math
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

def normalize_prediction(input_row,response):
    data=parse_json_object(response)
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
