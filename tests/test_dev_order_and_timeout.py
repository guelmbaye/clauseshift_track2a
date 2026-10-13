import json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from src.cli import main
from src.core import load_jsonl,write_jsonl
from src.inference import chat_completion, InferenceError

class DevOrderTests(unittest.TestCase):
    def test_shortest_dev_first_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); inputs=root/'inputs'; cps=root/'cps'; preds=root/'preds'
            write_jsonl(inputs/'dev_de.jsonl',[
                {'id':'admin_de_long','text_a':'Long text ' * 30,'text_b':'Langer Text ' * 30},
                {'id':'admin_de_short','text_a':'Good .','text_b':'Gut .'},
                {'id':'admin_de_middle','text_a':'More long words today','text_b':'Mehr lange Worte heute'}])
            def stub(url, model,prompt,key,**kwargs):
                # Return based on the exact requested tokens, with punctuation preserved.
                a=json.loads(prompt.split('Sentence 1: ')[-1].split('\nSentence 2: ')[0])
                b=json.loads(prompt.split('Sentence 2: ')[-1].split('\nRespond')[0])
                return json.dumps({'sentence1':[[t,-1 if t=='.' else 5] for t in a],
                                   'sentence2':[[t,-1 if t=='.' else 5] for t in b]})
            with patch('src.cli.chat_completion',side_effect=stub),patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.publicai.co/v1'}):
                rc=main(['run','--input-dir',str(inputs),'--output-dir',str(preds),'--checkpoint-dir',str(cps),
                         '--lang','de','--dev-shortest-first','--max-examples','2'])
            self.assertEqual(rc,0)
            saved=list(load_jsonl(cps/'dev_de.checkpoint.jsonl'))
            self.assertEqual([r['id'] for r in saved],['admin_de_short','admin_de_middle'])
            self.assertFalse((preds/'CLAUSESHIFT_admin_de.jsonl.jsonl').exists())
            # Resume with same settings and confirm final output uses the canonical source order.
            with patch('src.cli.chat_completion',side_effect=stub),patch.dict(os.environ,{'LLM_NAME':'swiss-ai/apertus-v1.5-8b','LLM_BASE_URL':'https://api.publicai.co/v1'}):
                rc=main(['run','--input-dir',str(inputs),'--output-dir',str(preds),'--checkpoint-dir',str(cps),
                         '--lang','de','--dev-shortest-first'])
            self.assertEqual(rc,0)
            final=list(load_jsonl(preds/'CLAUSESHIFT_admin_de.jsonl.jsonl'))
            self.assertEqual([r['id'] for r in final],['admin_de_long','admin_de_short','admin_de_middle'])

    def test_no_partial_final(self):
        self.assertEqual(main(['run','--split','final','--dev-shortest-first','--confirm-final',
                               '--frozen-commit','0123456789abcdef']),1)
        self.assertEqual(main(['run','--split','final','--lang','de','--confirm-final',
                               '--frozen-commit','0123456789abcdef']),1)

    def test_504_identifies_real_provider(self):
        exc=HTTPError('https://api.publicai.co/v1/chat/completions',504,'Gateway Timeout',{},None)
        with patch('src.inference.urlopen',side_effect=exc):
            with self.assertRaises(InferenceError) as cm:
                chat_completion('https://api.publicai.co/v1','swiss-ai/apertus-v1.5-8b','hi',retries=0)
        self.assertIn('api.publicai.co',str(cm.exception))
        self.assertNotIn('CSCS',str(cm.exception))

if __name__=='__main__':unittest.main()
