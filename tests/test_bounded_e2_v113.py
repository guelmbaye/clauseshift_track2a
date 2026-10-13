import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import write_jsonl
from src.e2 import bounded_request, fingerprint, run
from tests.test_e2 import args


class TestBoundedE2(unittest.TestCase):
    def test_bound_and_fairness(self):
        a,b=bounded_request(set(range(30)),set(range(30)),12)
        self.assertEqual((len(a),len(b)),(6,6))
        self.assertEqual(sorted(a),list(range(6)))
        self.assertEqual(sorted(b),list(range(6)))
        self.assertEqual(bounded_request(set(),{1,3,5},2),(set(),{1,3}))
        self.assertEqual(bounded_request({2,4},{1},0),({2,4},{1}))
        self.assertEqual(bounded_request({2,4},{1},20),({2,4},{1}))

    def test_split_exact_recovery_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row={'id':'id_1','text_a':'one two three four five six seven eight nine ten',
                 'text_b':'eins zwei drei vier funf sechs sieben acht neun zehn'}
            write_jsonl(root/'data'/'dev_de.jsonl',[row])
            o=args(root, max_scores_per_call=4, max_calls=2,
                   max_attempts_per_unit=10)
            calls=[]
            def infer(url,model,prompt,key,**kw):
                payload=json.loads(prompt.split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
                a,b=payload['requested_a'],payload['requested_b']
                calls.append((a,b))
                self.assertLessEqual(len(a)+len(b),4)
                return json.dumps({'a':{str(i):5 for i in a},'b':{str(i):0 for i in b}})
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.example/v1'}
            with patch.dict(os.environ,env):
                self.assertEqual(run(o,infer=infer),0)
                self.assertEqual(len(calls),2)
                o.max_calls=20
                self.assertEqual(run(o,infer=infer),0)
                self.assertEqual(len(calls),6)  # two cores each side, 4 request budget
                self.assertEqual(run(o,infer=infer),0)
                self.assertEqual(len(calls),6)  # no replay after completion
            from src.e2 import read_document_updates,completed_document
            from src.e2_alignment import save_or_load_plan
            plan=save_or_load_plan(root/'plans'/'dev_de_id_1.json',row,o)
            sig=fingerprint(o,env['LLM_NAME'],env['LLM_BASE_URL'])
            saved=read_document_updates(root/'checkpoints'/'dev_de_e2.checkpoint.jsonl',sig,row,plan)
            pred=completed_document(row,plan,saved)
            self.assertEqual(pred['labels_a'],[0]*10)
            self.assertEqual(pred['labels_b'],[1]*10)

    def test_budget_config_signature_isolation(self):
        root=Path('.')
        o=args(root)
        model='swiss-ai/apertus-v1.5-8b';endpoint='https://api.example/v1'
        before=fingerprint(o,model,endpoint)
        o.max_scores_per_call=0
        self.assertEqual(before,fingerprint(o,model,endpoint))
        o.max_scores_per_call=36
        self.assertNotEqual(before,fingerprint(o,model,endpoint))

    def test_offline_budget_punct(self):
        from tools.dev_e2_runtime_budget import counts
        row={'text_a':'one , two .','text_b':'eins , zwei .'}
        self.assertEqual(counts(row),4)

if __name__=='__main__':unittest.main()
