import re
from foundation.LLM.LLM_chat import llm_server
prompt_config = {
            "stop": None,
            "temperature": 0.3,
            "maxTokens": 1000,
            'model': "gpt-4-1106-preview",
        }
def extract_text_between_tags(file_path, tag_id):
    with open(file_path, 'r') as file:
        content = file.read()
        pattern = r'id:{} LLM cycle reflect prompt:(.*?)LLM cycle reflect response:(.*?)id:'.format(tag_id)
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            prompt_str = match[0].lstrip("\[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': ")
            prompt_str = prompt_str.lstrip('"').rstrip('\n').rstrip(' ').rstrip("}]").rstrip('"')
            # print(prompt_str)
            prompt_str += "There are five rating levels for you to evaluate the consistency score. 5: Completely consistent with the personal portrait. 4: Basically consistent, individual decisions are not consistent, but it is acceptable because the degree of violation is not deep. 3: Basically consistent, individual decisions are inconsistent and completely contrary to personal portrait. 2: Most decisions are inconsistent. 1: Completely inconsistent and contradictory. You should offer your rating number after tag [Rating:]."
            response, _, actual_prompt = llm_server(
                {"prompt": prompt_str, **prompt_config}, mode = 'chat'
            )
            print(str(actual_prompt) + response)


file_path = "../gpt3.5_05_23_cycle_70_6.log"
tag_id = 1  # Replace with the desired tag ID
extract_text_between_tags(file_path, tag_id)