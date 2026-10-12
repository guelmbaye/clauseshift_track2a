"""OpenAI-compatible Apertus inference; intentionally no dependency on OpenAI SDK."""
import json, time, os
from urllib.request import Request, urlopen
from urllib.error import HTTPError,URLError

class InferenceError(RuntimeError): pass

def chat_completion(base_url, model, prompt,api_key='',timeout=52,max_tokens=4096,retries=1):
    if not base_url:raise InferenceError('LLM_BASE_URL required')
    if 'apertus' not in model.lower():raise InferenceError('Refusing non-Apertus primary model')
    url=base_url.rstrip('/')+'/chat/completions'
    payload={'model':model,'messages':[{'role':'user','content':prompt}],
             'temperature':0,'max_tokens':max_tokens,'stream':False}
    headers={'Content-Type':'application/json', 'User-Agent':os.environ.get('LLM_USER_AGENT','CLAUSESHIFT/1.0')}
    if api_key:headers['Authorization']='Bearer '+api_key
    request=Request(url,json.dumps(payload).encode('utf-8'),headers,method='POST')
    for attempt in range(retries+1):
        try:
            with urlopen(request,timeout=timeout) as r:body=json.load(r)
            content=body['choices'][0]['message']['content']
            if not isinstance(content,str):raise InferenceError('Unexpected response content')
            return content
        except HTTPError as e:
            if e.code==504:raise InferenceError('CSCS 504/60s: persistent timeout. Do not blindly retry; change strategy or serving.') from e
            if e.code in (408,429,500,502,503) and attempt<retries:
                time.sleep(min(4,2**attempt));continue
            raise InferenceError(f'HTTP {e.code}: inference failed') from e
        except (URLError,TimeoutError) as e:
            if attempt<retries:
                time.sleep(min(4,2**attempt));continue
            raise InferenceError(f'Inference connection/timeout failure: {e}') from e
        except (KeyError,IndexError,ValueError,TypeError) as e:raise InferenceError('Malformed chat completion payload') from e
