import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.core import PredictionError, load_jsonl, write_jsonl
from src.e1i import encode_indexed_prompt, parse_partial, run, unique_json


def args(p, **kw):
    base = dict(lang='de', doc_id='', input_dir=p/'input',checkpoint_dir=p/'cp',
        output_dir=p/'out', diagnostics_dir=p/'diag', chunk_tokens=8, context=3, max_tokens=512, timeout=75,
        retries=0, max_calls=1,max_documents=1,max_attempts_per_chunk=3,
        shortest_first=False)
    base.update(kw)
    return SimpleNamespace(**base)


class E1ITests(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        for s in ['{"a":{"0":1,"0":2},"b":{}}', '{"a":{},"a":{},"b":{}}']:
            with self.assertRaises(PredictionError):
                unique_json(s)

    def test_reject_extra_invalid_and_punctuation(self):
        tests = ['{"a":{"8":5},"b":{}}',
                 '{"a":{"1":5},"b":{}}', # position 1 punctuation
                 '{"a":{"0":true},"b":{}}',
                 '{"a":{"0":6},"b":{}}',
                 '{"a":{},"b":{}}',
                 '{"a":[5],"b":{}}']
        for x in tests:
            with self.subTest(x=x), self.assertRaises(PredictionError):
                parse_partial(x,['word','.'],['mot'],{0},{0})

    def test_one_valid_partial_map(self):
        a,b = parse_partial('{"a":{"0":4},"b":{}}', ['hi','.','there'], ['bonjour'], {0,2}, {0})
        self.assertEqual(a,{0:4.0})
        self.assertEqual(b,{})

    def test_prompt_explicit_indices(self):
        p=encode_indexed_prompt(['hello','.','world'], ['bonjour','.','monde'], (0,3),(0,3),3,{0,2},{0,2})
        self.assertIn('"requested_a":[0,2]',p)
        self.assertIn('"i":2',p)

    def test_missing_scores_checkpoint_and_selective_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            row={'id':'admin_de_27','text_a':'one two . four five six seven eight nine ten',
                'text_b':'eins zwei . vier fuenf sechs sieben acht neun zehn'}
            write_jsonl(p/'input'/'dev_de.jsonl',[row])
            events=[]
            def fake(url,model,prompt,key,**kwargs):
                payload=json.loads(prompt.split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
                a=payload['requested_a'];b=payload['requested_b']
                events.append((a,b))
                # On first call, deliberately omit last requested english index.
                if len(events)==1:
                    a=a[:-1]
                return json.dumps({'a':{str(i):5 for i in a},'b':{str(i):0 for i in b}})
            environ={'LLM_NAME':'swiss-ai/apertus-v1.5-8b', 'LLM_BASE_URL':'https://api.example/v1'}
            opts=args(p)
            with patch.dict(os.environ,environ):
                self.assertEqual(run(opts,infer=fake),0)
                saved=list(load_jsonl(p/'cp'/'dev_de_indexed.checkpoint.jsonl'))
                self.assertEqual(len(saved),1)
                self.assertEqual(len(events),1)
                opts.max_calls=3
                self.assertEqual(run(opts,infer=fake),0)
                self.assertEqual(len(events),3)  # repair first chunk + second chunk
                self.assertEqual(events[1][0],[4])  # only missing index in first 5-token chunk
                self.assertEqual(events[1][1],[])
                pred=list(load_jsonl(p/'out'/'CLAUSESHIFT_admin_de.jsonl.jsonl'))
                self.assertEqual(len(pred),1)
                self.assertEqual(len(pred[0]['labels_a']),10)
                self.assertEqual(pred[0]['labels_a'][2],-1)
                self.assertEqual(pred[0]['labels_b'][0],1)
                self.assertEqual(run(opts,infer=fake),0)
                self.assertEqual(len(events),3)

    def test_missing_never_padded_on_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            write_jsonl(p/'input'/'dev_de.jsonl', [{'id':'id_1','text_a':'one two three four five six seven eight',
                                                    'text_b':'eins zwei drei vier fuenf sechs sieben acht'}])
            opts=args(p, max_calls=0, max_attempts_per_chunk=1)
            calls=[]
            def fake(url,model,prompt,key,**kw):
                calls.append(1)
                return '{"a":{"0":5},"b":{"0":4}}'
            with patch.dict(os.environ,{'LLM_NAME':'Apertus-8B','LLM_BASE_URL':'https://api.example/v1'}):
                self.assertEqual(run(opts,infer=fake),1)
            self.assertEqual(len(calls),1)
            self.assertFalse((p/'out'/'CLAUSESHIFT_admin_de.jsonl.jsonl').exists())

    def test_bad_output_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            write_jsonl(p/'input'/'dev_de.jsonl',[{'id':'admin_de_0','text_a':'one two', 'text_b':'eins zwei'}])
            opts=args(p)
            with patch.dict(os.environ, {'LLM_NAME':'Apertus-8B','LLM_BASE_URL':'https://api.example/v1'}):
                self.assertEqual(run(opts,infer=lambda *a,**kw:'{"a":{"55":3},"b":{}}'),1)
            self.assertFalse((p/'cp'/'dev_de_indexed.checkpoint.jsonl').read_text().strip())
            self.assertEqual(len(list((p/'diag').glob('*_raw.txt'))),1)


if __name__=='__main__': unittest.main()
