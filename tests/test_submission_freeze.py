import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import submission

class FrozenRunnerTest(unittest.TestCase):
    def test_no_manifest_preflight_does_not_access_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(submission.cli,'do_fetch_final',side_effect=AssertionError('test accessed')):
                with self.assertRaisesRegex(ValueError,'Missing frozen manifest'):
                    submission.run_final(Path(tmp)/'missing.json')

    def test_freeze_and_preflight_detect_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frozen.json'
            model='swiss-ai/apertus-v1.5-8b'
            data=submission.freeze(path,'a'*40,'b'*40,model)
            self.assertEqual(submission.preflight(path)['code_sha256'],data['code_sha256'])
            data['code_sha256']='0'*64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'code changed'):
                submission.preflight(path)

    def test_manifest_guard_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frozen.json'
            submission.freeze(path,'a'*40,'b'*40,'swiss-ai/apertus-v1.5-8b')
            with patch.dict(os.environ,{'LLM_NAME':'not-apertus','LLM_BASE_URL':'http://example','LLM_API_KEY':'x'}):
                with self.assertRaisesRegex(ValueError,'LLM_NAME'):
                    submission.preflight(path,require_secrets=True)

    def test_final_runner_orders_calls_after_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'freeze.json'
            submission.freeze(manifest,'a'*40,'b'*40,'swiss-ai/apertus-v1.5-8b')
            actions=[]
            with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'http://mock/v1','LLM_API_KEY':'fake'}):
                with patch.object(submission.cli,'do_fetch_final',side_effect=lambda _:actions.append('fetch')):
                    with patch.object(submission.cli,'do_run',side_effect=lambda _:actions.append('infer')):
                        with patch.object(submission.cli,'do_validate',side_effect=lambda _:actions.append('validate')):
                            original_hash = submission.sha_file
                            with patch.object(submission,'sha_file',side_effect=lambda p: original_hash(p) if not Path(p).name.startswith('CLAUSESHIFT_') else '1'*64):
                                submission.run_final(manifest)
            self.assertEqual(actions,['fetch','infer','validate'])

    def test_no_false_complete_if_inference_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'freeze.json'
            submission.freeze(manifest,'a'*40,'b'*40,'swiss-ai/apertus-v1.5-8b')
            with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'http://mock/v1','LLM_API_KEY':'fake'}):
                with patch.object(submission.cli,'do_fetch_final'):
                    with patch.object(submission.cli,'do_run'):
                        with patch.object(submission.cli,'do_validate',side_effect=FileNotFoundError('partial')):
                            with self.assertRaisesRegex(FileNotFoundError,'partial'):
                                submission.run_final(manifest)

if __name__=='__main__':unittest.main()
