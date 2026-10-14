"""Synthetic tests only: no held-out benchmark reads or API calls."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src import e2, submission
from src.core import load_jsonl, write_jsonl


def fake_args(root, **changes):
    base = dict(
        lang='all', doc_id='', input_dir=root/'inputs', output_dir=root/'out',
        checkpoint_dir=root/'cp', plan_dir=root/'plans', diagnostics_dir=root/'diagnostics',
        alignment='proportional', prompt='fewshot',
        embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        embedding_revision='0'*40, chunk_tokens=48, max_unit_tokens=48,
        similarity_threshold=0.46, merge_penalty=0.06, coverage_weight=0.50,
        anchor_bonus=0.35, context=20, max_tokens=1536,
        max_scores_per_call=96, timeout=75, retries=0,
        max_calls=0, max_documents=0, max_attempts_per_unit=8,
        shortest_first=False, plan_only=False, frozen_commit='a'*40,
        _frozen_authorized=True,
    )
    base.update(changes)
    return SimpleNamespace(**base)


class TestE2SingleSubmission(TestCase):
    def test_final_refused_before_authorization(self):
        with TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                e2.run(fake_args(Path(td), _frozen_authorized=False), split='final')

    def test_final_rejects_partial_runs(self):
        with TemporaryDirectory() as td:
            for bad in ({'doc_id':'sample'}, {'max_calls':1}, {'lang':'de'},
                        {'max_documents':1}, {'shortest_first':True}, {'plan_only':True}):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'Final mode'):
                    e2.run(fake_args(Path(td),**bad), split='final')

    def test_dev_fingerprint_unchanged_and_final_isolated(self):
        with TemporaryDirectory() as td:
            args=fake_args(Path(td))
            dev=e2.fingerprint(args,'swiss-ai/apertus-v1.5-8b','https://api.publicai.co/v1')
            self.assertEqual(dev,e2.fingerprint(args,'swiss-ai/apertus-v1.5-8b',
                                                 'https://api.publicai.co/v1','dev'))
            self.assertNotEqual(dev,e2.fingerprint(args,'swiss-ai/apertus-v1.5-8b',
                                                   'https://api.publicai.co/v1','final'))

    def test_synthetic_final_complete_and_replay(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            (root/'inputs').mkdir()
            # Deliberately reverse ordering to prove original-order export.
            for lang in ('de','fr','it'):
                rows=[{'id':f'admin_{lang}_z','text_a':'Alpha beta .',
                       'text_b':'Alfa beta .'},
                      {'id':f'admin_{lang}_a','text_a':'One two .',
                       'text_b':'Uno due .'}]
                write_jsonl(root/'inputs'/f'final_{lang}.jsonl',rows)
            args=fake_args(root)
            calls=[]
            def infer(endpoint,model,prompt,key,**kw):
                # Same schema as the validated single-unit E2 parser.
                payload=json.loads(prompt.split('Payload:\n',1)[1].split('\nReturn JSON only:',1)[0])
                a={str(i):5 for i in payload['requested_a']}
                b={str(i):5 for i in payload['requested_b']}
                calls.append(1)
                return json.dumps({'a':a,'b':b})
            with patch.dict(os.environ, {'LLM_NAME':'swiss-ai/apertus-v1.5-8b',
                                          'LLM_BASE_URL':'https://api.publicai.co/v1',
                                          'LLM_API_KEY':'synthetic'}, clear=False):
                result=e2.run(args,infer=infer,split='final')
                self.assertEqual(result,0)
                first=len(calls)
                self.assertEqual(first,6)
                e2.run(args,infer=infer,split='final')
                self.assertEqual(len(calls),first)
            for lang in ('de','fr','it'):
                path=root/'out'/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'
                exported=list(load_jsonl(path))
                self.assertEqual([r['id'] for r in exported], [f'admin_{lang}_z',f'admin_{lang}_a'])
                for item in exported:
                    self.assertEqual(item['labels_a'],[0.0,0.0,-1.0])
                    self.assertEqual(item['labels_b'],[0.0,0.0,-1.0])
                self.assertTrue((root/'plans'/f'final_dev_{lang}_admin_{lang}_z.json').exists())
                self.assertFalse((root/'plans'/f'dev_{lang}_admin_{lang}_z.json').exists())

    def test_single_freeze_preflight_and_limits(self):
        with TemporaryDirectory() as td:
            path=Path(td)/'frozen.json'
            payload=submission.freeze(path,'a'*40,'b'*40,
                         'swiss-ai/apertus-v1.5-8b',method='E2-SINGLE',
                         embedding_revision='c'*40,batch_cap=96,units_per_batch=3,
                         workers=8, max_tokens=1536)
            self.assertEqual(payload['schema'],3)
            self.assertEqual(payload['e2']['max_units_per_batch'],1)
            self.assertEqual(payload['e2']['workers'],1)
            self.assertEqual(submission.preflight(path)['method'],'E2-SINGLE')
            payload['e2']['workers']=2
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'Single-unit'):
                submission.preflight(path)

    def test_single_submission_driver_routes_to_unbatched_e2(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            frozen=submission.freeze(root/'frozen.json','a'*40,'b'*40,
                         'swiss-ai/apertus-v1.5-8b',method='E2-SINGLE',
                         embedding_revision='c'*40,batch_cap=96, max_tokens=1536)
            with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b',
                                         'LLM_BASE_URL':'https://api.publicai.co/v1',
                                         'LLM_API_KEY':'synthetic'}), \
                 patch.object(submission,'preflight', return_value=frozen), \
                 patch.object(submission.cli,'do_fetch_final') as fetch, \
                 patch.object(submission.cli,'do_validate') as validate, \
                 patch.object(submission.e2 if hasattr(submission,'e2') else e2, 'run', return_value=0) as e2_run, \
                 patch.object(submission, 'sha_file', return_value='1'*64):
                submission.run_final(root/'frozen.json')
                fetch.assert_called_once()
                validate.assert_called_once()
                args=e2_run.call_args.args[0]
                self.assertEqual(e2_run.call_args.kwargs['split'],'final')
                self.assertEqual(args.lang,'all')
                self.assertEqual(args.prompt,'fewshot')
                self.assertEqual(args.alignment,'semantic_coverage')
                self.assertEqual(args.max_scores_per_call,96)
                self.assertTrue(args._frozen_authorized)
                self.assertEqual(args.max_calls,0)
