import json,os,threading,tempfile,unittest
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from unittest.mock import patch
from src.cli import main
from src.core import load_jsonl,write_jsonl

class MockApertus(BaseHTTPRequestHandler):
    def do_POST(self):
        data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if 'Apertus' not in data['model']:self.send_response(400);self.end_headers();return
        # This fixture is NOT real Apertus and does not imply benchmark performance.
        content=json.dumps({'sentence1':[['Good',5],['.',-1]],'sentence2':[['Gut',5],['.',-1]]})
        raw=json.dumps({'choices':[{'message':{'content':content}}]}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
        self.end_headers();self.wfile.write(raw)
    def log_message(self,*args):pass

class RunnerTests(unittest.TestCase):
    def test_integration_three_languages(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);inputs=t/'inputs';preds=t/'out';cps=t/'cp'
            for lang in ('de','fr','it'):
                write_jsonl(inputs/f'dev_{lang}.jsonl',[{'id':f'admin_{lang}_0','text_a':'Good .','text_b':'Gut .'}])
            server=HTTPServer(('127.0.0.1',0),MockApertus)
            th=threading.Thread(target=server.serve_forever,daemon=True);th.start()
            try:
                with patch.dict(os.environ,{'LLM_NAME':'swiss-ai/Apertus-v1.5-8B','LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1'}):
                    rc=main(['run','--input-dir',str(inputs),'--output-dir',str(preds),'--checkpoint-dir',str(cps),'--max-tokens','100'])
                self.assertEqual(rc,0)
                self.assertEqual(main(['validate','--input-dir',str(inputs),'--output-dir',str(preds)]),0)
                for lang in ('de','fr','it'):
                    self.assertEqual(len(list(load_jsonl(preds/f'CLAUSESHIFT_admin_{lang}.jsonl.jsonl'))),1)
            finally:server.shutdown();server.server_close();th.join(timeout=2)

if __name__=='__main__':unittest.main()
