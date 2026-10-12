import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.core import (parse_json_object,project_token_labels,encode_prompt,normalize_prediction,
                      validate_input_rows,valid_prediction,PredictionError)
from src.cli import main as cli

class CoreTests(unittest.TestCase):
    def setUp(self):self.row={'id':'admin_de_0','text_a':'Good .','text_b':'Gut .'}
    def test_projection(self):
        d={'sentence1':[['Good',5],['.',-1]],'sentence2':[['Gut',3],['.',-1]]}
        p=normalize_prediction(self.row,json.dumps(d))
        self.assertEqual(p['labels_a'],[0.0,-1.0]);self.assertAlmostEqual(p['labels_b'][0],0.4)
        valid_prediction(p,self.row)
    def test_token_mismatch_fails(self):
        with self.assertRaises(PredictionError):project_token_labels({'sentence1':[['Bad',5]]},['Good'],'sentence1')
    def test_missing_tokens_fails(self):
        with self.assertRaises(PredictionError):project_token_labels({'sentence1':[]},['Good'],'sentence1')
    def test_nonjson_fails(self):
        with self.assertRaises(PredictionError):parse_json_object('plain old explanation')
    def test_prompt_tokens(self):self.assertIn('["Good", "."]',encode_prompt(self.row['text_a'],self.row['text_b']))
    def test_input_duplicate_fails(self):
        with self.assertRaises(ValueError):validate_input_rows([self.row,self.row])
    def test_test_split_not_exposed(self):
        with self.assertRaises(SystemExit) as e:cli(['score-dev','--upstream-root','/tmp','--split','test'])
        self.assertEqual(e.exception.code,2)
    def test_final_requires_freeze(self):
        self.assertEqual(cli(['prepare','--split','final','--upstream-root','/missing']),1)
    def test_strict_wrong_length(self):
        with self.assertRaises(ValueError):valid_prediction({'id':'admin_de_0','text_a':'Good .','text_b':'Gut .','labels_a':[0.1],'labels_b':[0,-1]},self.row)

if __name__=='__main__':unittest.main()
