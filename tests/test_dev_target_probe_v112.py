import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import main
from src.core import load_jsonl, write_jsonl
from tools.dev_e0_feasibility import report


class V112DevTargetProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.input_dir=self.root/'inputs'
        self.cps=self.root/'checkpoint'
        self.preds=self.root/'preds'
        self.rows=[
            {'id':'admin_de_1','text_a':'One .','text_b':'Eins .'},
            {'id':'admin_de_2','text_a':'Second sentence !','text_b':'Zweiter Satz !'},
            {'id':'admin_de_3','text_a':'Third .','text_b':'Dritte .'},
        ]
        write_jsonl(self.input_dir/'dev_de.jsonl',self.rows)
        for lang in ('fr','it'):
            write_jsonl(self.input_dir/f'dev_{lang}.jsonl',
                        [{'id':f'admin_{lang}_1','text_a':'A .','text_b':'B .'}])

    @staticmethod
    def fake_llm(url,model,prompt,key,**kwargs):
        a=json.loads(prompt.split('Sentence 1: ')[-1].split('\nSentence 2: ')[0])
        b=json.loads(prompt.split('Sentence 2: ')[-1].split('\nRespond')[0])
        return json.dumps({'sentence1':[[t,-1 if t in ('.','!') else 5] for t in a],
                           'sentence2':[[t,-1 if t in ('.','!') else 5] for t in b]})

    def run_one(self, doc_id):
        with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.publicai.co/v1'}):
            with patch('src.cli.chat_completion',side_effect=self.fake_llm) as llm:
                rc=main(['run','--split','dev','--lang','de','--doc-id',doc_id,
                         '--input-dir',str(self.input_dir),'--checkpoint-dir',str(self.cps),
                         '--output-dir',str(self.preds)])
                return rc,llm.call_count

    def test_target_single_id_saves_only_target_then_preserves_order(self):
        self.assertEqual(self.run_one('admin_de_2'),(0,1))
        self.assertEqual([x['id'] for x in load_jsonl(self.cps/'dev_de.checkpoint.jsonl')],['admin_de_2'])
        self.assertFalse((self.preds/'CLAUSESHIFT_admin_de.jsonl.jsonl').exists())
        self.assertEqual(self.run_one('admin_de_2'),(0,0))
        self.assertEqual(self.run_one('admin_de_3'),(0,1))
        self.assertEqual(self.run_one('admin_de_1'),(0,1))
        saved=list(load_jsonl(self.preds/'CLAUSESHIFT_admin_de.jsonl.jsonl'))
        self.assertEqual([r['id'] for r in saved],[r['id'] for r in self.rows])

    def test_invalid_doc_id_does_not_call_provider(self):
        self.assertEqual(self.run_one('does-not-exist'),(1,0))

    def test_final_rejects_target_before_io(self):
        with patch('src.cli.chat_completion',side_effect=AssertionError('provider called')):
            self.assertEqual(main(['run','--split','final','--confirm-final','--frozen-commit','a'*40,
                                   '--lang','de','--doc-id','admin_de_1']),1)

    def test_only_dev_and_single_language_and_no_reordering(self):
        self.assertEqual(main(['run','--lang','all','--doc-id','admin_de_1']),1)
        self.assertEqual(main(['run','--lang','de','--doc-id','admin_de_1','--smoke-shortest']),1)

    def test_offline_histogram_without_gold_or_test(self):
        result=report(self.input_dir,self.cps)
        self.assertEqual(result['de']['all'],3)
        self.assertEqual(result['de']['completed'],0)
        self.assertTrue(result['de']['buckets'])


if __name__=='__main__':unittest.main()
