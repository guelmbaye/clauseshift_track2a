"""Offline synthetic tests; no Public AI, Hugging Face, or held-out data."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.core import PredictionError, load_jsonl, write_jsonl
from src.e2_batch import candidate_batches,parse_batch,run_dataset,signature,batch_prompt
from src.e2_alignment import make_plan,save_or_load_plan
from src.e2 import read_document_updates, completed_document
from src import submission
from tests.test_e2 import args as e2args


def opts(root,**kwargs):
    o=e2args(root,lang='de',doc_id='id_1',alignment='proportional',
             prompt='fewshot',chunk_tokens=8,max_unit_tokens=8,
             max_scores_per_call=12,max_units_per_batch=3,workers=1,
             launch_interval=0.0,max_tokens=1536,max_calls=0,
             max_attempts_per_unit=8)
    o.frozen_commit=''
    for k,v in kwargs.items():setattr(o,k,v)
    return o


def response(url,model,prompt,key,**kw):
    payload=json.loads(prompt.split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
    return json.dumps({'units':{
        str(unit['unit']):{
            'a':{str(j):5 for j in unit['requested_a']},
            'b':{str(j):0 for j in unit['requested_b']}
        } for unit in payload['units']}})


class BatchV114(unittest.TestCase):
    def test_units_merged_and_complete(self):
        row={'id':'id_1', 'text_a':' '.join('one two three four five six seven eight'.split()*3),
             'text_b':' '.join('eins zwei drei vier funf sechs sieben acht'.split()*3)}
        o=opts(Path('.'),max_scores_per_call=32)
        plan=make_plan(row,o)
        batches=candidate_batches(row,plan,{},32,3)
        self.assertEqual(len(plan['units']),3)
        self.assertEqual(len(batches),2)
        self.assertEqual([j for j,_,_ in batches[0]],[0,1])
        self.assertEqual(sum(len(a)+len(b) for bat in batches for _,a,b in bat),48)
        self.assertTrue(all(sum(len(a)+len(b) for _,a,b in bat)<=32 for bat in batches))

    def test_multi_unit_real_checkpoint_from_one_mock_call(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row={'id':'id_1',
                 'text_a':'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen',
                 'text_b':'eins zwei drei vier funf sechs sieben acht neun zehn elf zwolf dreizehn vierzehn funfzehn sechzehn'}
            write_jsonl(root/'data'/'dev_de.jsonl',[row])
            o=opts(root,max_scores_per_call=32,max_units_per_batch=2,max_calls=0,doc_id="",max_documents=0)
            seen=[]
            def infer(*a,**kw):
                payload=json.loads(a[2].split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
                seen.append(payload)
                return response(*a,**kw)
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b',
                 'LLM_BASE_URL':'https://api.example/v1','LLM_API_KEY':'dummy'}
            with patch.dict(os.environ,env):
                self.assertEqual(run_dataset(o,infer=infer),0)
            self.assertEqual(len(seen),1)
            self.assertEqual(len(seen[0]['units']),2)
            self.assertEqual(len(list(load_jsonl(root/'checkpoints'/'dev_de_e2batch.checkpoint.jsonl'))),2)
            self.assertEqual(len(list(load_jsonl(root/'output'/'CLAUSESHIFT_admin_de.jsonl.jsonl'))),1)

    def test_atomic_validation_no_missing_or_extra_indices(self):
        row={'id':'id_1','text_a':'one two three four','text_b':'eins zwei drei vier'}
        o=opts(Path('.'))
        plan=make_plan(row,o)
        batch=candidate_batches(row,plan,{},12,3)[0]
        valid=response(None,None,batch_prompt(o,row,plan,batch),None)
        self.assertEqual(len(parse_batch(valid,row,plan,batch)),1)
        bad=json.loads(valid)
        del bad['units']['0']['a']['0']
        with self.assertRaisesRegex(PredictionError,'Incomplete batch'):
            parse_batch(json.dumps(bad),row,plan,batch)
        bad=json.loads(valid)
        bad['units']['0']['a']['99']=5
        with self.assertRaises(PredictionError):parse_batch(json.dumps(bad),row,plan,batch)
        with self.assertRaises(PredictionError):parse_batch(valid[:-1],row,plan,batch)
        dup=valid.replace('"0": 5','"0": 5, "0": 3',1)
        with self.assertRaises(PredictionError):parse_batch(dup,row,plan,batch)

    def test_exact_checkpoint_resume_and_output_reorder(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            data=[{'id':'id_1','text_a':'one two three four five six seven eight',
                   'text_b':'eins zwei drei vier funf sechs sieben acht'},
                  {'id':'id_2','text_a':'alpha beta gamma delta',
                   'text_b':'alpha beta gamma delta'}]
            write_jsonl(root/'data'/'dev_de.jsonl',data)
            o=opts(root,max_scores_per_call=8,max_calls=1,max_documents=0,doc_id='',
                   shortest_first=True)
            calls=[]
            def infer(*a,**kw):
                calls.append(1)
                return response(*a,**kw)
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.example/v1','LLM_API_KEY':'dummy'}
            with patch.dict(os.environ,env):
                self.assertEqual(run_dataset(o,infer=infer),0)
                self.assertEqual(len(calls),1)
                self.assertFalse((root/'output'/'CLAUSESHIFT_admin_de.jsonl.jsonl').exists())
                o.max_calls=10
                self.assertEqual(run_dataset(o,infer=infer),0)
                total=len(calls)
                self.assertEqual(run_dataset(o,infer=infer),0)
                self.assertEqual(len(calls),total)
                result=list(load_jsonl(root/'output'/'CLAUSESHIFT_admin_de.jsonl.jsonl'))
            self.assertEqual([r['id'] for r in result],['id_1','id_2'])
            self.assertEqual(result[0]['labels_a'],[0.0]*8)
            self.assertEqual(result[0]['labels_b'],[1.0]*8)

    def test_parallel_failure_retains_valid_batch_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row={'id':'id_1','text_a':'one two three four five six seven eight',
                 'text_b':'eins zwei drei vier funf sechs sieben acht neun zehn elf zwolf dreizehn vierzehn funfzehn sechzehn'}
            row['text_a']+=' nine ten eleven twelve thirteen fourteen fifteen sixteen'
            write_jsonl(root/'data'/'dev_de.jsonl',[row])
            o=opts(root,max_scores_per_call=16,workers=2,max_calls=0)
            def infer(*args,**kw):
                data=json.loads(args[2].split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
                if data['units'][0]['unit']==1:return '{"units":'
                return response(*args,**kw)
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.example/v1','LLM_API_KEY':'dummy'}
            with patch.dict(os.environ,env):
                self.assertEqual(run_dataset(o,infer=infer),1)
                cp=root/'checkpoints'/'dev_de_e2batch.checkpoint.jsonl'
                saved=list(load_jsonl(cp))
                self.assertEqual(len(saved),1)
                self.assertEqual(saved[0]['unit_index'],0)
                self.assertFalse((root/'output'/'CLAUSESHIFT_admin_de.jsonl.jsonl').exists())
                self.assertEqual(run_dataset(o,infer=response),0)
                self.assertEqual(len(list(load_jsonl(cp))),2)

    def test_final_guard_and_freeze_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            o=opts(root)
            with self.assertRaisesRegex(ValueError,'Final/test inputs are forbidden'):
                run_dataset(o,split='final',infer=response)
            mf=root/'freeze.json'
            data=submission.freeze(mf,'a'*40,'b'*40,'swiss-ai/apertus-v1.5-8b',1536,
                                   method='E2-BATCH',embedding_revision='e'*40,
                                   batch_cap=96,units_per_batch=3,workers=2)
            self.assertEqual(submission.preflight(mf)['e2']['workers'],2)
            data['e2']['max_scores_per_call']=200
            mf.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Invalid frozen E2 settings'):
                submission.preflight(mf)

    def test_final_submission_routes_to_batch_after_preflight(self):
        from unittest.mock import patch
        from src import e2_batch
        with tempfile.TemporaryDirectory() as d:
            mf=Path(d)/'freeze.json'
            submission.freeze(mf,'a'*40,'b'*40,'swiss-ai/apertus-v1.5-8b',1536,
                              method='E2-BATCH',embedding_revision='e'*40)
            observed=[]
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b',
                 'LLM_BASE_URL':'http://mock/v1','LLM_API_KEY':'dummy'}
            with patch.dict(os.environ,env):
                with patch.object(submission.cli,'do_fetch_final',side_effect=lambda _:observed.append('fetch')):
                    def spy(arg,**kw):
                        observed.append(('batch',kw['split'],arg._frozen_authorized,arg.max_scores_per_call))
                        return 0
                    with patch.object(e2_batch,'run_dataset',side_effect=spy):
                        with patch.object(submission.cli,'do_validate',side_effect=lambda _:observed.append('validate')):
                            original=submission.sha_file
                            def sha_or_fake(path):
                                if Path(path).name.startswith('CLAUSESHIFT_admin_'):return '1'*64
                                return original(path)
                            with patch.object(submission,'sha_file',side_effect=sha_or_fake):
                                submission.run_final(mf)
            self.assertEqual(observed[0],'fetch')
            self.assertEqual(observed[1],('batch','final',True,96))
            self.assertEqual(observed[2],'validate')

    def test_method_signature_isolated(self):
        o=opts(Path('.'))
        x=signature(o,'swiss-ai/apertus-v1.5-8b','https://example/v1')
        o.workers=4
        self.assertEqual(x,signature(o,'swiss-ai/apertus-v1.5-8b','https://example/v1'))
        o.max_scores_per_call=96
        self.assertNotEqual(x,signature(o,'swiss-ai/apertus-v1.5-8b','https://example/v1'))

if __name__=='__main__':unittest.main()
