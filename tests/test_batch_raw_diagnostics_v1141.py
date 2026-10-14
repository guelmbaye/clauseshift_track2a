"""Mock-only tests for fail-closed diagnostics, with no real API calls."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.core import write_jsonl, load_jsonl
from src.e2_batch import run_dataset
from tests.test_e2_batch_v114 import opts, response
from tools.inspect_batch_raw import analyze_raw


class TestBatchDiagnostics(unittest.TestCase):
    def test_inspector_understands_two_objects(self):
        x=analyze_raw('{"units":{"0":{}}}{"units":{"1":{}}}')
        self.assertTrue(x['valid_prefix'])
        self.assertEqual(x['extra_type'],'second_json_value')
        self.assertEqual(x['prefix_unit_keys'],['0'])
        self.assertEqual(x['second_unit_keys'],['1'])

    def test_inspector_understands_commentary(self):
        x=analyze_raw('{"units":{}} More scores below')
        self.assertEqual(x['extra_type'],'other_text_or_value')

    def test_invalid_batch_saves_raw_owner_only_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_jsonl(root/'data'/'dev_de.jsonl',
                        [{'id':'id_1','text_a':'one two three four five six seven eight',
                          'text_b':'eins zwei drei vier funf sechs sieben acht'}])
            args=opts(root,doc_id='id_1',max_calls=1,max_documents=1,
                      max_scores_per_call=16,diagnostics_dir=root/'diagnostics')
            def bad(*a,**kw):return response(*a,**kw)+'{"additional":true}'
            with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b',
                                        'LLM_BASE_URL':'https://mock.example/v1',
                                        'LLM_API_KEY':'dummy'}):
                self.assertEqual(run_dataset(args,infer=bad),1)
            raws=list((root/'diagnostics').glob('*.txt'))
            self.assertEqual(len(raws),1)
            self.assertEqual(raws[0].stat().st_mode & 0o777,0o600)
            self.assertEqual(analyze_raw(raws[0].read_text())['extra_type'],'second_json_value')
            cp=root/'checkpoints'/'dev_de_e2batch.checkpoint.jsonl'
            self.assertFalse(cp.exists() and cp.read_text().strip())

    def test_valid_batch_does_not_save_raw(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_jsonl(root/'data'/'dev_de.jsonl',
                        [{'id':'id_1','text_a':'one two three four',
                          'text_b':'eins zwei drei vier'}])
            args=opts(root,doc_id='id_1',max_calls=1,max_documents=1,
                      diagnostics_dir=root/'diagnostics')
            with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b',
                                        'LLM_BASE_URL':'https://mock.example/v1',
                                        'LLM_API_KEY':'dummy'}):
                self.assertEqual(run_dataset(args,infer=response),0)
            self.assertFalse((root/'diagnostics').exists())


if __name__=='__main__':unittest.main()
