"""Local-only regression tests: batch examples agree with parser schema."""
import json
from pathlib import Path
import unittest
from src.core import PredictionError
from src.e2_alignment import make_plan
from src.e2_batch import batch_fewshot, batch_prompt, candidate_batches, parse_batch, signature
from tests.test_e2_batch_v114 import opts, response


class TestBatchFewshotSchema(unittest.TestCase):
    def test_three_original_examples_are_wrapped_in_batch_envelope(self):
        prompt=batch_fewshot()
        demos=[line[len('Output: '):] for line in prompt.splitlines()
               if line.startswith('Output: ')]
        self.assertEqual(len(demos), 3)
        for demo in demos:
            parsed=json.loads(demo)
            self.assertEqual(set(parsed), {'units'})
            self.assertEqual(set(parsed['units']), {'0'})
            self.assertEqual(set(parsed['units']['0']), {'a','b'})
        self.assertIn('Two-unit schema example', prompt)
        self.assertIn('Do not generate any root keys besides', prompt)

    def test_fewshot_prompt_and_mock_multi_unit_parser_still_agree(self):
        row={'id':'id_1', 'text_a':'one two three four five six seven eight nine ten eleven twelve',
             'text_b':'eins zwei drei vier funf sechs sieben acht neun zehn elf zwolf'}
        args=opts(Path('.'),max_scores_per_call=24,max_units_per_batch=3,
                  chunk_tokens=8,max_unit_tokens=8)
        plan=make_plan(row,args)
        batch=candidate_batches(row,plan,{},24,3)[0]
        prompt=batch_prompt(args,row,plan,batch)
        self.assertIn('Payload:\n',prompt)
        self.assertIn('Return JSON only:',prompt)
        self.assertTrue(parse_batch(response(None,None,prompt,None),row,plan,batch))

    def test_reject_extra_json_and_bad_root_even_when_prefix_is_valid(self):
        row={'id':'id_1','text_a':'one two three four','text_b':'eins zwei drei vier'}
        args=opts(Path('.'))
        plan=make_plan(row,args)
        batch=candidate_batches(row,plan,{},12,3)[0]
        good=response(None,None,batch_prompt(args,row,plan,batch),None)
        bad=json.loads(good)
        bad['1']={'a':{},'b':{}}
        with self.assertRaises(PredictionError):
            parse_batch(json.dumps(bad),row,plan,batch)
        with self.assertRaises(PredictionError):
            parse_batch(good+'   some extra text',row,plan,batch)

    def test_old_checkpoints_are_version_isolated(self):
        args=opts(Path('.'))
        current=signature(args,'swiss-ai/apertus-v1.5-8b','https://example/v1')
        self.assertNotEqual(current,'b59a30fe4e62')


if __name__=='__main__': unittest.main()
