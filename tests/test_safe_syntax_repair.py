import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core import normalize_prediction, remove_one_extraneous_rparen, PredictionError, write_jsonl, load_jsonl
from src.cli import main, digest_settings
from tools.recover_dev_e0pj import recover, sigs


class TestSafeSyntaxRepair(unittest.TestCase):
    def setUp(self):
        self.row={'id':'admin_de_27','text_a':'the energy sector )',
                  'text_b':'der Energiesektor )'}
        self.output={'sentence1': [['the',5],['energy',5],['sector',5],[')',-1]],
                     'sentence2':[['der',5],['Energiesektor',5],[')',-1]]}
        self.valid=json.dumps(self.output,ensure_ascii=False)
        self.bad=self.valid.replace('["sector", 5],', '["sector", 5]),')

    def test_exact_repair(self):
        self.assertIn('["sector", 5]),',self.bad)
        with self.assertRaisesRegex(PredictionError, 'Non-JSON'):
            normalize_prediction(self.row,self.bad,punctuation_repair=True)
        record=[]
        pred=normalize_prediction(self.row,self.bad,punctuation_repair=True,syntax_repair=True,syntax_log=record)
        self.assertEqual(pred['labels_a'],[0.,0.,0.,-1.])
        self.assertEqual(pred['labels_b'],[0.,0.,-1.])
        self.assertEqual(len(record),1)
        self.assertTrue(record[0]['changed'])
        fixed,meta=remove_one_extraneous_rparen(self.bad)
        self.assertEqual(json.loads(fixed), self.output)
        self.assertEqual(len(fixed),len(self.bad)-1)

    def test_valid_json_unchanged_with_paren_in_token(self):
        fixed,meta=remove_one_extraneous_rparen(self.valid)
        self.assertEqual(fixed,self.valid)
        self.assertFalse(meta['changed'])

    def test_reject_different_error(self):
        other=self.valid.replace('["sector", 5],','["sector", 5]')
        with self.assertRaises(PredictionError):remove_one_extraneous_rparen(other)

    def test_reject_two_stray_chars(self):
        twice=self.bad.replace('["energy", 5],','["energy", 5]),')
        with self.assertRaises(PredictionError):remove_one_extraneous_rparen(twice)

    def test_reject_changed_nonpunctuation_token_even_if_json_fix(self):
        bad=self.bad.replace('["energy", 5]', '["energies", 5]')
        with self.assertRaisesRegex(PredictionError, 'semantic token drift'):
            normalize_prediction(self.row,bad,punctuation_repair=True,syntax_repair=True)

    def test_reject_out_of_range_score_even_if_syntax_fix(self):
        bad=self.bad.replace('["energy", 5]', '["energy", 19]')
        with self.assertRaisesRegex(PredictionError, 'invalid score'):
            normalize_prediction(self.row,bad,punctuation_repair=True,syntax_repair=True)

    def test_checkpoint_distinction(self):
        with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.publicai.co/v1'}):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                write_jsonl(root/'inputs'/'dev_de.jsonl',[self.row])
                with patch('src.cli.chat_completion',return_value=self.bad):
                    rc=main(['run','--input-dir',str(root/'inputs'),
                             '--checkpoint-dir',str(root/'cp'), '--lang','de',
                             '--punctuation-repair','--syntax-repair',
                             '--max-examples','1'])
                self.assertEqual(rc,0)
                checkpoint=list(load_jsonl(root/'cp'/'dev_de.checkpoint.jsonl'))
                self.assertEqual(len(checkpoint),1)
                self.assertEqual(checkpoint[0]['_syntax_repair_log'][0]['kind'],'removed-stray-rparen')
                self.assertIn('syntax-repair-v1',checkpoint[0]['_postprocessing'])

if __name__=='__main__':unittest.main()

class TestOfflineRecovery(unittest.TestCase):
    def test_recovery_dryrun_and_write_separate(self):
        from argparse import Namespace
        from unittest.mock import patch
        from tools import recover_dev_e0pj as rec
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for lang in ('de','fr','it'):
                write_jsonl(root/'data'/'inputs'/f'dev_{lang}.jsonl',
                    [{'id':f'admin_{lang}_105','text_a':'Good .','text_b':'Gut .'}]
                    + ([{'id':'admin_de_27','text_a':'the energy sector )','text_b':'der Energiesektor )'}] if lang=='de' else []))
            prompt_file=root/'paper_exact.txt'
            prompt_file.write_text('s1={{ sentence1 }} s2={{ sentence2 }}')
            model='swiss-ai/apertus-v1.5-8b'; url='https://api.publicai.co/v1'
            old,new=sigs(model,prompt_file.read_text(),4096,url)
            src=root/'e0p';src.mkdir()
            for lang in ('de','fr','it'):
                write_jsonl(src/f'dev_{lang}.checkpoint.jsonl', [{'_signature':old,
                    'id':f'admin_{lang}_105','text_a':'Good .','text_b':'Gut .',
                    'labels_a':[0,-1],'labels_b':[0,-1]}])
            raw_json={'sentence1': [['the',5],['energy',5],['sector',5],[')',-1]],
                      'sentence2':[['der',5],['Energiesektor',5],[')',-1]]}
            raw=json.dumps(raw_json).replace('["sector", 5],', '["sector", 5]),')
            sha=hashlib.sha256(raw.encode()).hexdigest()
            raw_file=root/f'dev_de_admin_de_27_20261013T000000Z_{sha[:10]}_raw.txt'
            raw_file.write_text(raw)
            target=root/'e0pj'
            args=Namespace(source_checkpoints=src,destination_checkpoints=target,
                raw_file=raw_file,lang='de',doc_id='admin_de_27',prompt_file=prompt_file,
                max_tokens=4096,write=False)
            with patch.object(rec,'ROOT',root),patch.dict(os.environ,{'LLM_NAME':model,'LLM_BASE_URL':url}):
                rec.recover(args)
                self.assertFalse(target.exists())
                args.write=True
                rec.recover(args)
            self.assertEqual(len(list(load_jsonl(target/'dev_de.checkpoint.jsonl'))),2)
            self.assertEqual(len(list(load_jsonl(target/'dev_fr.checkpoint.jsonl'))),1)
            self.assertEqual(len(list(load_jsonl(src/'dev_de.checkpoint.jsonl'))),1)
            for lang in ('de','fr','it'):
                for row in load_jsonl(target/f'dev_{lang}.checkpoint.jsonl'):
                    self.assertEqual(row['_signature'],new)
