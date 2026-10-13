import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.core import load_jsonl, write_jsonl
from src.e2_alignment import make_plan, save_or_load_plan, validate_units, token_spans
from src.e2 import run, FEWSHOT


def args(root, **kwargs):
    opts=dict(lang='de',doc_id='id_1',input_dir=root/'data',checkpoint_dir=root/'checkpoints',
              plan_dir=root/'plans',output_dir=root/'output',diagnostics_dir=root/'diag',
              alignment='proportional',prompt='short',embedding_model='fake-model',
              embedding_revision='test-commit',max_unit_tokens=8,similarity_threshold=0.4,
              merge_penalty=0.08,coverage_weight=0.5,anchor_bonus=0.35,chunk_tokens=8,context=4,max_tokens=512,timeout=75,
              retries=0,max_calls=1,max_documents=1,max_attempts_per_unit=3,
              shortest_first=False,plan_only=False)
    opts.update(kwargs)
    return SimpleNamespace(**opts)

class E2Tests(unittest.TestCase):
    def test_sentence_spans_preserve_numeric_date(self):
        spans=token_spans('Last update 15 . 09 . 2020 . Links'.split(),20)
        self.assertEqual(spans[0],(0,8))
        self.assertEqual(spans[1],(8,9))

    def test_semantic_alignment_out_of_proportional_order_gaps(self):
        # Embedding stub: sentence vocab encodes a shared concept, and a DE-only extra unit.
        row={'id':'id_1','text_a':'apple banana . lemon orange .',
             'text_b':'apfel banane . ZUSATZ . zitrone orange .'}
        vecs={'apple':(1.,0.,0.),'apfel':(1.,0.,0.),
              'lemon':(0.,1.,0.),'zitrone':(0.,1.,0.),
              'ZUSATZ':(0.,0.,1.)}
        def encoder(sentences):
            return [vecs[next((token for token in vecs if token in t.split()),'ZUSATZ')]
                    for t in sentences]
        o=args(Path('.'), alignment='semantic',max_unit_tokens=8,chunk_tokens=8,
               similarity_threshold=0.5)
        plan=make_plan(row,o,encoder)
        validate_units(plan['units'],6,8,8)
        self.assertEqual([u['kind'] for u in plan['units']],['matched','unaligned_b','matched'])
        self.assertEqual(plan['units'][1]['rb'],[3,5])
        self.assertEqual(plan['units'][1]['ra'],[3,3])

    def test_corrupt_or_gapped_plan_rejected(self):
        with self.assertRaises(ValueError):
            validate_units([{'ra':[0,2],'rb':[0,2]}, {'ra':[3,4],'rb':[2,4]}],4,4,8)
        with self.assertRaises(ValueError):
            validate_units([{'ra':[0,9],'rb':[0,8]}],9,8,8)

    def test_plan_provenance_and_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row={'id':'id_1','text_a':'hello dear', 'text_b':'bonjour cher'}
            opts=args(root)
            path=root/'plan.json'
            save_or_load_plan(path,row,opts)
            plan=save_or_load_plan(path,row,opts)
            self.assertEqual(plan['id'],'id_1')
            plan['units'][0]['ra']=[1,2]
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):save_or_load_plan(path,row,opts)

    def test_checkpoint_resume_and_no_fabricated_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            # Proportional plan: 2 windows, 8 tokens per side.
            row={'id':'id_1','text_a':'one two three four five six seven eight nine ten',
                 'text_b':'eins zwei drei vier funf sechs sieben acht neun zehn'}
            write_jsonl(root/'data'/'dev_de.jsonl',[row])
            o=args(root)
            seen=[]
            def infer(url,model,prompt,key,**kw):
                payload=json.loads(prompt.split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
                a=payload['requested_a'];b=payload['requested_b']
                seen.append((a,b))
                if len(seen)==1:a=a[:-1]
                return json.dumps({'a':{str(i):5 for i in a},'b':{str(i):3 for i in b}})
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.example/v1'}
            with patch.dict(os.environ,env):
                self.assertEqual(run(o,infer=infer),0)
                self.assertEqual(len(seen),1)
                o.max_calls=10
                self.assertEqual(run(o,infer=infer),0)
                self.assertEqual(len(seen),3) # missing + second window
                self.assertEqual(seen[1],([4],[]))
                # Calling again does not bill any further requests.
                self.assertEqual(run(o,infer=infer),0)
                self.assertEqual(len(seen),3)
            from src.e2 import read_document_updates, completed_document, fingerprint
            plan=save_or_load_plan(root/'plans'/'dev_de_id_1.json',row,o)
            sig=fingerprint(o,env['LLM_NAME'],env['LLM_BASE_URL'])
            state=read_document_updates(root/'checkpoints'/'dev_de_e2.checkpoint.jsonl',sig,row,plan)
            result=completed_document(row,plan,state)
            self.assertEqual(result['labels_a'],[0.0]*10)
            self.assertEqual(result['labels_b'],[0.4]*10)

    def test_absent_scores_never_padded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row={'id':'id_1','text_a':'one two', 'text_b':'eins zwei'}
            write_jsonl(root/'data'/'dev_de.jsonl',[row])
            o=args(root,max_attempts_per_unit=1)
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.example/v1'}
            with patch.dict(os.environ,env):
                with self.assertRaises(ValueError):
                    run(o,infer=lambda *a,**kw:'{"a":{"0":5},"b":{"0":5}}')
            self.assertFalse((root/'output'/'CLAUSESHIFT_admin_de.jsonl.jsonl').exists())

    def test_fewshot_is_explicitly_adapted(self):
        self.assertIn('Example 3',FEWSHOT)
        self.assertIn('Output:',FEWSHOT)

if __name__=='__main__':unittest.main()

# Additional integration tests for semantic-planned inference, independently of HF/GPU.
class E2SemanticIntegration(unittest.TestCase):
    def test_semantic_plan_only_then_reuse_same_frozen_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row={'id':'id_1','text_a':'apple banana . lemon orange .',
                 'text_b':'apfel banane . ZUSATZ . zitrone orange .'}
            write_jsonl(root/'data'/'dev_de.jsonl',[row])
            o=args(root, alignment='semantic',max_unit_tokens=8,chunk_tokens=8,
                   similarity_threshold=0.5,plan_only=True)
            vecs={'apple':(1.,0.,0.),'apfel':(1.,0.,0.),
                  'lemon':(0.,1.,0.),'zitrone':(0.,1.,0.),'ZUSATZ':(0.,0.,1.)}
            encoder=lambda sentences:[vecs[next((x for x in vecs if x in t.split()),'ZUSATZ')]
                                      for t in sentences]
            self.assertEqual(run(o,embedder=encoder),0)
            saved=root/'plans'/'dev_de_id_1.json'
            self.assertTrue(saved.exists())
            o.plan_only=False
            logs=[]
            def fake(url,model,prompt,key,**kw):
                payload=json.loads(prompt.split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
                logs.append(payload)
                return json.dumps({'a':{str(i):5 for i in payload['requested_a']},
                                   'b':{str(i):2 for i in payload['requested_b']}})
            env={'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.example/v1'}
            with patch.dict(os.environ,env):
                o.max_calls=1
                self.assertEqual(run(o,infer=fake,embedder=lambda _:self.fail('should use frozen plan')),0)
                o.max_calls=20
                self.assertEqual(run(o,infer=fake,embedder=lambda _:self.fail('should use frozen plan')),0)
            self.assertEqual(len(logs),3)
            self.assertTrue(any(not x['requested_a'] for x in logs))  # DE-only segment
            from src.e2 import completed_document,read_document_updates,fingerprint
            from src.e2_alignment import save_or_load_plan
            plan=save_or_load_plan(saved,row,o)
            saved_scores=read_document_updates(root/'checkpoints'/'dev_de_e2.checkpoint.jsonl',
                               fingerprint(o,env['LLM_NAME'],env['LLM_BASE_URL']),row,plan)
            pred=completed_document(row,plan,saved_scores)
            self.assertEqual(len(pred['labels_a']),6)
            self.assertEqual(len(pred['labels_b']),8)
            self.assertEqual(pred['labels_b'][3],0.6)
            self.assertEqual(pred['labels_a'][2],-1)
