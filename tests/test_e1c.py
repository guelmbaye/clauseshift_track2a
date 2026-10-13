import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from src.e1c import chunk_ranges,score_arrays,make_prompt,run,collect_one,read_chunks
from src.core import PredictionError,write_jsonl,load_jsonl

class E1CTests(unittest.TestCase):
    def test_chunk_cover_both_languages(self):
        for a,b in [(128,101),(725,190),(1,20),(100,100)]:
            chunks=chunk_ranges(a,b,80)
            self.assertEqual(sum(z[0][1]-z[0][0] for z in chunks),a)
            self.assertEqual(sum(z[1][1]-z[1][0] for z in chunks),b)
            self.assertEqual(chunks[0][0][0],0)
            self.assertEqual(chunks[-1][1][1],b)
    def test_numeric_output_and_punctuation(self):
        x,y=score_arrays('{"scores_a":[5,-1,0],"scores_b":[0,-1]}',
                         ['hello',',','world'],['bonjour','.'])
        self.assertEqual(x,[0.0,-1.0,1.0]);self.assertEqual(y,[1.0,-1.0])
    def test_malformed_arrays_fail(self):
        samples=['{"scores_a":[5],"scores_b":[1]}',
                 '{"scores_a":[5,0],"scores_b":[1],"extra":0}',
                 '{"scores_a":[5,true],"scores_b":[1]}']
        for s in samples:
            with self.assertRaises(PredictionError):
                score_arrays(s,['a','b'],['c'])
    def test_dev_integration_resume_order(self):
        with tempfile.TemporaryDirectory() as path:
            p=Path(path); inputdir=p/'data';cpdir=p/'check';outdir=p/'out'
            records=[{'id':'admin_de_3','text_a':'one two three four five six seven eight nine ten',
                      'text_b':'eins zwei drei vier fuenf sechs sieben acht neun zehn'},
                     {'id':'admin_de_2','text_a':'Alpha Beta .','text_b':'Alfa Beta .'}]
            write_jsonl(inputdir/'dev_de.jsonl',records)
            args=SimpleNamespace(lang='de',input_dir=inputdir,checkpoint_dir=cpdir,
                output_dir=outdir,chunk_tokens=8,context=2,max_tokens=768,timeout=75,
                retries=0,max_calls=1,max_documents=0,shortest_first=True,doc_id='')
            calls=[]
            def fake(url,model,prompt,key,**kw):
                d=json.loads(prompt.split('JSON input:\n')[1].split('\nReturn JSON:')[0])
                calls.append(d)
                aa=d['english']['core'];bb=d['other_language']['core']
                return json.dumps({'scores_a':[-1 if t=='.' else 5 for t in aa],
                                   'scores_b':[-1 if t=='.' else 5 for t in bb]})
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.example.org/v1','LLM_API_KEY':'fake'}
            with patch.dict(os.environ,env):
                self.assertEqual(run(args,infer=fake),0)
                self.assertEqual(len(calls),1)
                args.max_calls=100;args.max_documents=0
                self.assertEqual(run(args,infer=fake),0)
                pred=list(load_jsonl(outdir/'CLAUSESHIFT_admin_de.jsonl.jsonl'))
                self.assertEqual([r['id'] for r in pred],[r['id'] for r in records])
                self.assertEqual(pred[1]['labels_a'],[0,0,-1])
                self.assertEqual(len(calls),3)  # shortest 1 chunk, longer 2 chunks
                self.assertEqual(run(args,infer=fake),0)
                self.assertEqual(len(calls),3)
    def test_fail_closed_invalid_responses(self):
        with tempfile.TemporaryDirectory() as path:
            p=Path(path);write_jsonl(p/'dev_de.jsonl',[{'id':'admin_de_0','text_a':'A B', 'text_b':'C D'}])
            args=SimpleNamespace(lang='de',input_dir=p,checkpoint_dir=p/'cp',output_dir=p/'output',
                    chunk_tokens=8,context=2,max_tokens=768,timeout=75,retries=0,max_calls=1,
                    max_documents=1,shortest_first=False,doc_id='')
            with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://example.org/v1'}):
                self.assertEqual(run(args,infer=lambda *a,**kw:'{"scores_a":[5],"scores_b":[5]}'),1)
            self.assertFalse((p/'cp'/'dev_de_chunks.checkpoint.jsonl').read_text().strip())

if __name__=='__main__':unittest.main()
