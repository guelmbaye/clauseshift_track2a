"""Conservative restoration of one lost typographic double-quote at token boundary."""
import hashlib
import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from src.core import PredictionError, normalize_prediction, restore_punctuation_only, write_jsonl, load_jsonl
from src.cli import main
from tools import recover_dev_e0pj as rec

class BoundaryQuoteTests(unittest.TestCase):
    def setUp(self):
        self.row={'id':'admin_de_27', 'text_a':'the energy sector )',
                  'text_b':'Der Energiebereich» liefert Überblick .'}
        self.output={'sentence1': [['the',5],['energy',5],['sector',4],[')',-1]],
                     'sentence2':[['Der',5], ['Energiebereich',4], ['liefert',3], ['Überblick',2], ['.',-1]]}
        self.raw=json.dumps(self.output,ensure_ascii=False)
        self.raw_bad_syntax=self.raw.replace('["sector", 4],','["sector", 4]),')

    def test_strict_mode_rejects_unquoted_word(self):
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            normalize_prediction(self.row,self.raw,punctuation_repair=True)

    def test_expected_quote_restored_with_exact_score(self):
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            normalize_prediction(self.row,self.raw_bad_syntax, punctuation_repair=True,syntax_repair=True)
        repair_log, syntax_log=[],[]
        pred=normalize_prediction(self.row,self.raw_bad_syntax,
            punctuation_repair=True, syntax_repair=True,boundary_quote_repair=True,
            repair_log=repair_log,syntax_log=syntax_log)
        self.assertEqual([round(v, 5) for v in pred['labels_b']],[0.0,0.2,0.4,0.6,-1.0])
        self.assertEqual(len(repair_log[1]['boundary_quote_repairs']),1)
        self.assertEqual(repair_log[1]['boundary_quote_repairs'][0]['expected'],'Energiebereich»')
        self.assertEqual(syntax_log[0]['kind'],'removed-stray-rparen')

    def test_reject_changed_word_even_with_quote_option(self):
        obj=json.loads(self.raw)
        obj['sentence2'][1][0]='Energiebereiche'
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            normalize_prediction(self.row,json.dumps(obj),punctuation_repair=True,boundary_quote_repair=True)

    def test_reject_missing_apostrophe_inside_word(self):
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            restore_punctuation_only(["l'administration"],[["ladministration",3]],'sentence2',True)

    def test_reject_missing_semantic_parenthesis_and_comma(self):
        for expected, got in [('Energiebereich)', 'Energiebereich'),('3,5','35'), ('well-being','wellbeing'),('Zurich?','Zurich')]:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(PredictionError,'semantic token drift'):
                    restore_punctuation_only([expected],[[got,4]],'sentence2',True)

    def test_reject_two_missing_quote_chars(self):
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            restore_punctuation_only(['«Wort»'],[['Wort',4]],'sentence2',True)

    def test_reject_wrong_score(self):
        with self.assertRaisesRegex(PredictionError, 'punctuation score'):
            restore_punctuation_only(['Energiebereich»'],[['Energiebereich',-1]],'sentence2',True)

    def test_new_signature_isolated(self):
        with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.publicai.co/v1'}):
            with tempfile.TemporaryDirectory() as td:
                td=Path(td)
                write_jsonl(td/'inputs'/'dev_de.jsonl',[self.row])
                with patch('src.cli.chat_completion',return_value=self.raw_bad_syntax):
                    rc=main(['run','--input-dir',str(td/'inputs'), '--checkpoint-dir',str(td/'cp'),
                             '--lang','de','--punctuation-repair','--syntax-repair',
                             '--boundary-quote-repair','--max-examples','1'])
                self.assertEqual(rc,0)
                record=list(load_jsonl(td/'cp'/'dev_de.checkpoint.jsonl'))[0]
                self.assertIn('boundary-quote-repair-v1',record['_postprocessing'])
                self.assertAlmostEqual(record['labels_b'][1],.2)
                with patch('src.cli.chat_completion',return_value=self.raw_bad_syntax):
                    rc=main(['run','--input-dir',str(td/'inputs'), '--checkpoint-dir',str(td/'cp'),
                             '--lang','de','--punctuation-repair','--syntax-repair','--max-examples','1'])
                self.assertEqual(rc,1)

    def test_offline_recover_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for lang in ('de','fr','it'):
                rows=[{'id':f'admin_{lang}_105','text_a':'Good .','text_b':'Gut .'}]
                if lang=='de':rows.append(self.row)
                write_jsonl(root/'data'/'inputs'/f'dev_{lang}.jsonl',rows)
            prompt='s1={{ sentence1 }} s2={{ sentence2 }}'
            prompt_file=root/'paper_exact.txt';prompt_file.write_text(prompt)
            model='swiss-ai/apertus-v1.5-8b';base='https://api.publicai.co/v1'
            old,new=rec.sigs(model,prompt,4096,base,boundary_quote_repair=True)
            source=root/'e0p';source.mkdir()
            for lang in ('de','fr','it'):
                write_jsonl(source/f'dev_{lang}.checkpoint.jsonl',[{
                    '_signature':old, 'id':f'admin_{lang}_105','text_a':'Good .','text_b':'Gut .',
                    'labels_a':[0,-1],'labels_b':[0,-1]}])
            sha=hashlib.sha256(self.raw_bad_syntax.encode()).hexdigest()
            raw_file=root/f'dev_de_admin_de_27_20261013T063000Z_{sha[:10]}_raw.txt'
            raw_file.write_text(self.raw_bad_syntax)
            target=root/'e0pjq'
            args=Namespace(source_checkpoints=source,destination_checkpoints=target,
                           raw_file=raw_file,lang='de',doc_id='admin_de_27',
                           prompt_file=prompt_file,max_tokens=4096,write=False,boundary_quote_repair=True)
            with patch.object(rec,'ROOT',root),patch.dict(os.environ,{'LLM_NAME':model,'LLM_BASE_URL':base}):
                rec.recover(args)
                self.assertFalse(target.exists())
                args.write=True
                rec.recover(args)
            data=list(load_jsonl(target/'dev_de.checkpoint.jsonl'))
            self.assertEqual(len(data),2)
            self.assertEqual(data[1]['_signature'],new)
            self.assertAlmostEqual(data[1]['labels_b'][1],.2)
            self.assertEqual(len(list(load_jsonl(source/'dev_de.checkpoint.jsonl'))),1)

if __name__=='__main__':unittest.main()
