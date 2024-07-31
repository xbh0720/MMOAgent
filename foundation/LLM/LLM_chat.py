
import requests
import os
import gzip
import json as json
from base64 import b64decode, b64encode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

url_completion = ''
url_chat = ''
headers = {
    'Access-Key': 'yours',
    'Access-Secret': 'yours',
    'Content-Type': 'application/json',
    'Accept-Encoding': 'identity',
}
url_llama = 'http://localhost:11434/api/chat'
llama_data_chat = {
    "model": "llama3",
    "messages": [
        {
            "role": "user",
            "content": "why is the sky blue?"
        }
    ],
    "stream":False,
}
llama_headers = {'Content-Type': 'application/json'}

llm_data_completion = {
    "prompt": "《魔法巴黎》\n\n故事发生在巴黎的一个最繁华的夜晚。穿越着微醺的气息，一群优雅的艺术家、富丽堂皇的服装设计师，以及一位英俊的艺术家让整个夜晚洋溢着神奇般的气息。\n\n这天，维克洛让他的恋人嘉莉进入了他的艺术领域。面对复杂的巴黎街头，他空灵而神秘的写着：“只要心中有梦想，",
    # "model": "text-davinci-003",
    # text-davinci-003|text-ada-001|text-babbage-001|text-curie-001|davinci|text-davinci-001|text-davinci-002|gpt-3.5-turbo-instruct.*
    "model": "gpt-3.5-turbo-instruct", # "gpt-3.5-turbo-instruct",
    "maxTokens": 50,
    "temperature": 1,
    "topP": 1,
    "stop": None,
    "presencePenalty": 0,
    "frequencyPenalty": 0
}
llm_data_chat = {
    "messages": [
        {
          "role": "system",
          "content": "In a search system."
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "\n\nHello, how can I help you？"
        },
        {
            "role": "user",
            "content": "Do you know new bing？"
        }
    ],
    "model": "gpt-3.5-turbo", # "gpt-3.5-turbo-1106", #"gpt-4-0613", 
    "maxTokens": 50,
    "temperature": 1,
    "topP": 1,
    "stop": None,
    "presencePenalty": 0,
    "frequencyPenalty": 0
}
def retry_strategy():
    retry = Retry(
        total=5, 
        status_forcelist=[429, 500, 502, 503, 504], 
        # method_whitelist=["HEAD", "GET", "POST"], 
        backoff_factor=1 
    )
    adapter = HTTPAdapter(max_retries=retry)
    return adapter

def post_request(url, headers, json_data, timeout=50000):
    session = requests.Session()
    session.mount("https://", retry_strategy())
    try:
        response = session.post(url, headers=headers, json=json_data, timeout=timeout)
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Error Connecting: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error: {timeout_err}")  
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")
    return None


def llm_completion2chat(prompt):
    '''return {
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": "Please follow the instruction to finish the task."
            }
        ]
    }
    '''
    '''{
                "role": "system",
                "content": 'You are playing a role as game player in a mmo game.'
            },'''
    
    return {
        "messages": [       
            {
                "role": "system",
                "content": 'You are a helpful assistant.'
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }


def llm_server(req={}, mode='chat'):
    model  = req.get('model', 'gpt')
    if "gpt" in model:
        llm_data = llm_data_chat if mode == 'chat' else llm_data_completion
        url = url_chat if mode == 'chat' else url_completion
        if mode=='chat':
            req = {**req, **llm_completion2chat(req['prompt'])}
            del req['prompt']
        data_raw = {**llm_data, **req}
        response = post_request(url, headers, data_raw)
        try:
            if mode == 'chat':
                return response['detail']['choices'][0]['message']['content'], response, req['messages']
            else:
                return response['detail']['choices'][0]['text'], response, req['prompt']
        except Exception:
            print(response)
    elif "llama" in model:
        llm_data = llama_data_chat
        url = url_llama
        if mode=='chat':
            req = {**req, **llm_completion2chat(req['prompt'])}
            del req['prompt']
        data_raw = {**llm_data, **req}
        # print(data_raw)
        response = post_request(url, llama_headers, data_raw)
        try:
            print(response.keys())
            print(response['message']['content'])
            return response['message']['content'], response, req['messages']
        except:
            print(response)

if __name__ == '__main__':
    response = llm_server()
    print(response)
