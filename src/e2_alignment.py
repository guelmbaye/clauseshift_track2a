"""DEV-only language-aware contiguous alignment planning, without gold labels.

Semantic mode requires sentence-transformers. No silent lexical/proportional fallback.
Plans cover every original token once, including unmatched regions.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from .e1c import chunk_ranges

VERSION = 'e2-alignment-v1'
END = {'.', '!', '?', ';', '。', '！', '？'}


def token_spans(tokens, max_unit_tokens=20):
    if not 8 <= max_unit_tokens <= 128:
        raise ValueError('max-unit-tokens must be between 8 and 128')
    if not tokens:
        raise ValueError('Empty language side')
    spans = []
    start = 0
    for i, token in enumerate(tokens):
        # Do not split decimal numbers or Swiss-style spaced dates such as 15 . 09 . 2020.
        is_numeric_dot = (token == '.' and i > 0 and i+1 < len(tokens)
                          and tokens[i-1].isdigit() and tokens[i+1].isdigit())
        boundary = (token in END or (token.endswith(('.', '!', '?', ';')) and
                     len(token) > 1 and not re.match(r'^\d+(?:\.\d+)*\.$', token))) and not is_numeric_dot
        full = (i+1-start >= max_unit_tokens)
        if boundary or full:
            spans.append((start, i+1))
            start = i+1
    if start < len(tokens):
        spans.append((start, len(tokens)))
    return spans


def _embeddings(strings, model_name, revision):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('Semantic alignment requires: pip install sentence-transformers') from exc
    # Exact revision recommended for scientific reproducibility.
    model = SentenceTransformer(model_name, revision=revision, device='cpu')
    return model.encode(strings, normalize_embeddings=True, show_progress_bar=False,
                        batch_size=16)


def _vectors(tokens, spans, embedder):
    return [" ".join(tokens[start:stop]) for start, stop in spans]


def _cos(a,b):
    return sum(float(x)*float(y) for x,y in zip(a,b))


def _plan_semantic(a, b, *, max_unit, chunk_tokens, threshold, merge_penalty, embedder):
    sa=token_spans(a,max_unit)
    sb=token_spans(b,max_unit)
    # Precompute single and merged contiguous text representations.
    candidates_a={(i,k):(sa[i][0],sa[i+k-1][1]) for i in range(len(sa))
                  for k in (1,2) if i+k<=len(sa) and sa[i+k-1][1]-sa[i][0]<=chunk_tokens}
    candidates_b={(j,k):(sb[j][0],sb[j+k-1][1]) for j in range(len(sb))
                  for k in (1,2) if j+k<=len(sb) and sb[j+k-1][1]-sb[j][0]<=chunk_tokens}
    slots=[('a',i,k,rr) for (i,k),rr in candidates_a.items()] + [('b',j,k,rr) for (j,k),rr in candidates_b.items()]
    strings=[" ".join((a if side=='a' else b)[rr[0]:rr[1]]) for side,_,_,rr in slots]
    vectors=embedder(strings)
    if len(vectors)!=len(slots):raise ValueError('Encoder returned an incorrect number of vectors')
    vec={ (side,i,k):v for (side,i,k,_),v in zip(slots,vectors)}
    na,nb=len(sa),len(sb)
    neg=float('-inf')
    dp=[[neg]*(nb+1) for _ in range(na+1)]
    back={}
    dp[0][0]=0.0
    for i in range(na+1):
        for j in range(nb+1):
            score=dp[i][j]
            if score==neg:continue
            # Zero reward for an unmatched unit; never fill in a semantic score here.
            for ni,nj,kind,ki,kj,reward,sim in (
                (i+1,j,'unaligned_a',1,0,0.0,None),
                (i,j+1,'unaligned_b',0,1,0.0,None),
            ):
                if ni<=na and nj<=nb and score+reward>dp[ni][nj]+1e-10:
                    dp[ni][nj]=score+reward;back[(ni,nj)]=(i,j,kind,ki,kj,sim)
            for ki in (1,2):
                if (i,ki) not in candidates_a:continue
                for kj in (1,2):
                    if (j,kj) not in candidates_b:continue
                    sim=_cos(vec['a',i,ki],vec['b',j,kj])
                    if not math.isfinite(sim):raise ValueError('Nonfinite embedding similarity')
                    gain=sim-threshold-merge_penalty*(ki+kj-2)
                    if gain<=0:continue
                    ni,nj=i+ki,j+kj
                    if score+gain>dp[ni][nj]+1e-10:
                        dp[ni][nj]=score+gain;back[(ni,nj)]=(i,j,'matched',ki,kj,round(sim,5))
    if (na,nb) not in back:raise ValueError('Unable to build semantic alignment')
    i,j=na,nb
    units=[]
    while i or j:
        old_i,old_j,kind,ki,kj,sim=back[(i,j)]
        ra=(sa[old_i][0],sa[i-1][1]) if ki else ((sa[old_i][0] if old_i<na else len(a)),)*2
        rb=(sb[old_j][0],sb[j-1][1]) if kj else ((sb[old_j][0] if old_j<nb else len(b)),)*2
        units.append({'ra':list(ra),'rb':list(rb),'kind':kind,'similarity':sim})
        i,j=old_i,old_j
    units.reverse()
    return units


def _plan_proportional(a,b,chunk_tokens):
    return [{'ra':list(ra),'rb':list(rb),'kind':'proportional','similarity':None}
            for ra,rb in chunk_ranges(len(a),len(b),chunk_tokens)]


def validate_units(units,na,nb,chunk_tokens):
    i=j=0
    if not units:raise ValueError('Empty alignment plan')
    for k,unit in enumerate(units):
        ra,rb=unit.get('ra'),unit.get('rb')
        if (not isinstance(ra,list) or not isinstance(rb,list) or len(ra)!=2 or len(rb)!=2
            or type(ra[0]) is not int or type(ra[1]) is not int
            or type(rb[0]) is not int or type(rb[1]) is not int):
            raise ValueError(f'Malformed ranges in unit {k}')
        if ra[0]!=i or rb[0]!=j or ra[1]<i or rb[1]<j:
            raise ValueError(f'Gapped, reordered or overlapping alignment at unit {k}')
        if ra[1]==i and rb[1]==j:raise ValueError('Empty/empty alignment unit')
        if ra[1]-i>chunk_tokens or rb[1]-j>chunk_tokens:
            raise ValueError(f'Unit {k} exceeds configured output core limit')
        i,j=ra[1],rb[1]
    if i!=na or j!=nb:
        raise ValueError(f'Incomplete alignment coverage: {i}/{na}, {j}/{nb}')
    return True


def make_plan(row, args, embedder=None):
    a,b=row['text_a'].split(),row['text_b'].split()
    if args.alignment=='proportional':
        units=_plan_proportional(a,b,args.chunk_tokens)
    elif args.alignment=='semantic':
        if embedder is None:
            embedder=lambda texts:_embeddings(texts,args.embedding_model,args.embedding_revision)
        units=_plan_semantic(a,b,max_unit=args.max_unit_tokens,
              chunk_tokens=args.chunk_tokens,threshold=args.similarity_threshold,
              merge_penalty=args.merge_penalty,embedder=embedder)
    else:raise ValueError('Unknown alignment mode')
    validate_units(units,len(a),len(b),args.chunk_tokens)
    spec={key:getattr(args,key) for key in ('alignment','chunk_tokens','max_unit_tokens',
          'embedding_model','embedding_revision','similarity_threshold','merge_penalty')}
    result={'version':VERSION,'id':row['id'],
       'input_sha256':hashlib.sha256(json.dumps([a,b],ensure_ascii=False).encode()).hexdigest(),
       'settings':spec,'units':units}
    result['plan_sha256']=hashlib.sha256(json.dumps(units,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return result


def plan_path(plan_dir,lang,doc_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',doc_id):raise ValueError('Unsafe document ID')
    return Path(plan_dir)/f'dev_{lang}_{doc_id}.json'


def save_or_load_plan(path,row,args,embedder=None):
    path=Path(path)
    if path.exists():
        plan=json.loads(path.read_text(encoding='utf-8'))
        expected_settings={key:getattr(args,key) for key in ('alignment','chunk_tokens','max_unit_tokens',
              'embedding_model','embedding_revision','similarity_threshold','merge_penalty')}
        a,b=row['text_a'].split(),row['text_b'].split()
        if (plan.get('version')!=VERSION or plan.get('id')!=row['id'] or
            plan.get('settings')!=expected_settings or
            plan.get('input_sha256')!=hashlib.sha256(json.dumps([a,b],ensure_ascii=False).encode()).hexdigest() or
            plan.get('plan_sha256')!=hashlib.sha256(json.dumps(plan.get('units'),sort_keys=True,ensure_ascii=False).encode()).hexdigest()):
            raise ValueError(f'Alignment plan changed; use a fresh plan directory: {path}')
        validate_units(plan['units'],len(a),len(b),args.chunk_tokens)
        return plan
    plan=make_plan(row,args,embedder)
    path.parent.mkdir(parents=True,exist_ok=True)
    # No gold labels and no inference data in this artifact.
    with path.open('x',encoding='utf-8') as handle:
        json.dump(plan,handle,ensure_ascii=False,indent=2)
    return plan


def alignment_summary(plan,row):
    a,b=row['text_a'].split(),row['text_b'].split()
    miss_a=miss_b=0
    for u in plan['units']:
        if u['kind']=='unaligned_a':miss_a+=u['ra'][1]-u['ra'][0]
        if u['kind']=='unaligned_b':miss_b+=u['rb'][1]-u['rb'][0]
    return {'units':len(plan['units']),'unaligned_a':miss_a,'unaligned_b':miss_b,
            'unaligned_fraction':(miss_a+miss_b)/(len(a)+len(b))}
