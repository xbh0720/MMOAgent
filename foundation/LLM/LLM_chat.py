
import requests
import os
import gzip
import json as json
from base64 import b64decode, b64encode
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

url_completion = 'https://hangyan-openapi-gpt-proxy-11044-8080.apps.danlu.netease.com/proxy/hangyan/openapi/gpt/api/v2/text/completion'
url_chat = 'https://hangyan-openapi-gpt-proxy-11044-8080.apps.danlu.netease.com/proxy/hangyan/openapi/gpt/api/v2/text/chat'
headers = {
    'Access-Key': '845f3772d8fde3c131457028fd8de81706766ddfd2e5a62063429209ea5fc4f3',
    'Access-Secret': '11b59059e45dbd471f48e05be8428af8f65144743897ba0aed48e3f897b503d7',
    'Content-Type': 'application/json',
    'Accept-Encoding': 'identity',
    'projectId':'jmP1aez44k4Bo3EfI8RIJoC3NmHeGV2lDoKj',
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
          "content": "在一个搜索引擎中。"
        },
        {
            "role": "user",
            "content": "你好呀"
        },
        {
            "role": "assistant",
            "content": "\n\n你好，有什么可以帮助你的吗？"
        },
        {
            "role": "user",
            "content": "你认识new bing吗？"
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
        total=5, # 总共重试的次数
        status_forcelist=[429, 500, 502, 503, 504], # 指定哪些响应状态码需要重试
        # method_whitelist=["HEAD", "GET", "POST"], # 指定哪些请求方法需要重试
        backoff_factor=1 # 退避因子，指定等待时间的增加方式
    )
    adapter = HTTPAdapter(max_retries=retry)
    return adapter

def post_request(url, headers, json_data, timeout=50000):
    session = requests.Session()
    session.mount("https://", retry_strategy()) # 挂载重试策略到会话
    #出错了应该重新生成，这里重试3次，仍然直接return，gpt4明显选择的action更合理，但是速度也慢很多，
    try:
        response = session.post(url, headers=headers, json=json_data, timeout=timeout)
        response.raise_for_status()  # 如果响应不是 200，产生异常
        return response.json()  # 返回JSON响应
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # 打印HTTP错误
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Error Connecting: {conn_err}")  # 打印连接错误
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error: {timeout_err}")  # 打印超时错误
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")  # 打印其他错误
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
    # 发起POST请求的函数调用示例
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