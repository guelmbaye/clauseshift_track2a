import json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from src.cli import main
from src.core import write_jsonl

class InvalidRawDiagnosticsTests(unittest.TestCase):
    def test_malformed_json_is_saved_dev_and_not_checkpointed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); inputs=root/'inputs'; cp=root/'cp'
            write_jsonl(inputs/'dev_de.jsonl',[{'id':'admin_de_27','text_a':'Good .','text_b':'Gut .'}])
            malformed='{"sentence1": [["Good", 5] [".", -1]], "sentence2": [["Gut", 5], [".", -1]]}'
            with patch('src.cli.chat_completion',return_value=malformed), \
                 patch('src.cli.BASE',root), \
                 patch.dict(os.environ,{'LLM_BASE_URL':'https://api.publicai.co/v1', 'LLM_NAME':'swiss-ai/apertus-v1.5-8b'}):
                result=main(['run','--input-dir',str(inputs),'--checkpoint-dir',str(cp),'--lang','de','--max-examples','1'])
            self.assertEqual(result,1)
            raws=list((root/'out'/'diagnostics').glob('dev_de_admin_de_27_*_raw.txt'))
            self.assertEqual(len(raws),1)
            self.assertEqual(raws[0].read_text(),malformed)
            cp_file=cp/'dev_de.checkpoint.jsonl'
            self.assertFalse(cp_file.exists() and cp_file.stat().st_size)
    def test_valid_prediction_kept(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs=root/'inputs';cp=root/'cp'
            write_jsonl(inputs/'dev_de.jsonl',[{'id':'admin_de_27','text_a':'Good .','text_b':'Gut .'}])
            valid=json.dumps({'sentence1':[['Good',5],['.',-1]],'sentence2':[['Gut',5],['.',-1]]})
            with patch('src.cli.chat_completion',return_value=valid), \
                 patch('src.cli.BASE',root), \
                 patch.dict(os.environ,{'LLM_BASE_URL':'https://api.publicai.co/v1','LLM_NAME':'swiss-ai/apertus-v1.5-8b'}):
                result=main(['run','--input-dir',str(inputs),'--checkpoint-dir',str(cp),'--lang','de','--max-examples','1'])
            self.assertEqual(result,0)
            self.assertFalse((root/'out'/'diagnostics').exists())
            self.assertEqual(len((cp/'dev_de.checkpoint.jsonl').read_text().splitlines()),1)
