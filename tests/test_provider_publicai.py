import io, json, os, unittest
from unittest.mock import patch
from src.inference import chat_completion, InferenceError

class MockResponse:
    def __init__(self,payload): self.stream=io.BytesIO(json.dumps(payload).encode())
    def __enter__(self): return self.stream
    def __exit__(self,*args): return False

class TestPublicAI(unittest.TestCase):
    def test_lowercase_apertus_id_and_user_agent(self):
        payload={"choices":[{"message":{"content":"Apertus ready"}}]}
        captured=[]
        def fake_urlopen(request,timeout):
            captured.append(request)
            return MockResponse(payload)
        with patch('src.inference.urlopen',side_effect=fake_urlopen):
            ans=chat_completion('https://api.publicai.co/v1','swiss-ai/apertus-v1.5-8b','ping','dummy')
        self.assertEqual(ans,'Apertus ready')
        self.assertEqual(captured[0].get_header('User-agent'),'CLAUSESHIFT/1.0')
        self.assertEqual(captured[0].get_header('Authorization'),'Bearer dummy')
        self.assertEqual(json.loads(captured[0].data)['model'],'swiss-ai/apertus-v1.5-8b')

    def test_reject_non_apertus(self):
        with self.assertRaises(InferenceError):
            chat_completion('https://api.publicai.co/v1','some-other-model','ping')

if __name__ == '__main__': unittest.main()
