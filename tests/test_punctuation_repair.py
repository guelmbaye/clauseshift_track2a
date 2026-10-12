import json
import unittest
from src.core import (normalize_prediction, restore_punctuation_only,
                      is_punctuation_only, PredictionError)

class PunctuationOnlyTests(unittest.TestCase):
    def setUp(self):
        self.row={'id':'admin_fr_105', 'text_a':'One , two .',
                  'text_b':'les publications officielles et , dans une certaine mesure , le droit'}
        self.output={'sentence1':[['One',5],[',',-1],['two',4],['.',-1]],
                     'sentence2':[['les',5],['publications',4],['officielles',5],
                                  [',',-1],['et',4],['dans',5],['une',3],
                                  ['certaine',4],['mesure',4],['le',5],['droit',5]]}

    def test_diagnostic_reproduction_fail_closed_by_default(self):
        with self.assertRaisesRegex(PredictionError,'token entries required'):
            normalize_prediction(self.row,json.dumps(self.output))

    def test_repair_preserves_all_semantic_scores(self):
        log=[]
        p=normalize_prediction(self.row,json.dumps(self.output),True,log)
        self.assertEqual([round(v, 4) for v in p['labels_b']],[0,0.2,0,0.2,-1,0,0.4,0.2,0.2,-1,0,0])
        self.assertEqual([round(v, 4) for v in p['labels_a']],[0,-1,0.2,-1])
        self.assertTrue(log[1]['changed'])
        self.assertEqual(log[1]['source_punctuation'],2)
        self.assertEqual(log[1]['generated_punctuation'],1)

    def test_reject_missing_semantic_word(self):
        self.output['sentence2']=[x for x in self.output['sentence2'] if x[0]!='droit']
        with self.assertRaisesRegex(PredictionError,'semantic token count changed'):
            normalize_prediction(self.row,json.dumps(self.output),True)

    def test_reject_modified_semantic_word(self):
        self.output['sentence2'][3]=['yet',2]
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            normalize_prediction(self.row,json.dumps(self.output),True)

    def test_reject_nonpunctuation_minus_one(self):
        self.output['sentence2'][0]=['les',-1]
        with self.assertRaisesRegex(PredictionError,'punctuation score'):
            normalize_prediction(self.row,json.dumps(self.output),True)

    def test_reject_nan(self):
        self.output['sentence2'][0]=['les',float('nan')]
        with self.assertRaisesRegex(PredictionError,'invalid score'):
            normalize_prediction(self.row,json.dumps(self.output),True)

    def test_punctuation_mixed_token_not_rewritten(self):
        self.assertTrue(is_punctuation_only(','))
        self.assertTrue(is_punctuation_only('...'))
        self.assertFalse(is_punctuation_only("l'administration"))
        self.assertFalse(is_punctuation_only('3,5'))
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            restore_punctuation_only(["l'administration"],[['ladministration',4]],'sentence1')

    def test_ordered_correct_prediction_unchanged(self):
        expected=['x',',','y']
        got=[['x',5],[',',-1],['y',2]]
        fixed,info=restore_punctuation_only(expected,got,'sentence1')
        self.assertEqual(fixed,got)
        self.assertFalse(info['changed'])

    def test_reject_semantic_swap(self):
        with self.assertRaisesRegex(PredictionError,'semantic token drift'):
            restore_punctuation_only(['alpha','beta',','],[['beta',4],['alpha',4]],'sentence1')

if __name__=='__main__': unittest.main()
