import time
from typing import Any, Optional , Union
import re
from loguru import logger

from cirrus.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

from cirrus.configs.config import DEFAULT_MAX_RETRIES

import json
# from test_call_llm import test_call_qwen3_max,test_call_llm
# from independent_llm_service_v3 import LLMService
from cirrus.llm import call_llm

def check_content_format(content:str) -> bool:
    '''
    匹配输出格式是否符合要求 如果输出格式不符合要求 可能需要重新匹配  不然设置报错error
    '''
    #pattern = r'^\s*<thinking>(?P<thought>.*?)</thinking>\s*<action\s+name=["\']?(?P<name>.*?)["\']?>(?P<action>.*?)</action>\s*$'
    pattern = r'\s*<action\s+name=["\']?(?P<name>.*?)["\']?>(?P<action>.*?)</action>\s*$'
    return bool(re.match(pattern, content, re.DOTALL))


def to_llm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of odemessages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = message.tool_calls
            
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


# def generate_by_local_Qwen(
#     model: str,
#     messages: list[Message],
#     tools: Optional[list[any]] = None,
#     tool_choice: Optional[str] = None,
#     proxy_url: str = "http://172.16.115.209:8765",
#     timeout: int = 60,
#     max_retries: int = 3,
#     **kwargs: Any,
# ) -> Union[UserMessage , AssistantMessage]:
#     pass

#     url = f"{proxy_url}/api/call_robot"
#     tools = [tool.openai_schema for tool in tools] if tools else None
#     messages = to_llm_messages(messages)


#     #### headers
#     headers = {"Content-Type": "application/json"}
#     success = False
#     for attempt in range(max_retries):
#         result , token_usage , time_usage= test_call_llm(messages,model)
#         #print(result)
#         # result = response.json()
#         # #print(result)
#         # if result['success'] == True:
#         time.sleep(1)
#         try:

#             if check_content_format(result):
#                 message = action_xml_to_message(result)
#                 success = True
#                 message.cost = time_usage
#                 message.usage = token_usage
#                 return message,success
#             elif isinstance(result,str) and result:
#                 message  = AssistantMessage(role='assistant',content=result)
#                 success = True
#                 message.cost = time_usage
#                 message.usage = token_usage
#                 return message,success
#             else:
#                 print('输出格式错误')
#                 time.sleep(5)
#                 continue
#         except:
#             pass

#     return result,success


#     #print(messages[-1])
#     #print(response)

# def generate_by_LLMService(    model: str,
#                                 messages: list[Message],
#                                 tools: Optional[list[any]] = None,
#                                 timeout: int = 60,
#                                 max_retries: int = 3,):
    
#     llm =LLMService()
    
#     messages = to_llm_messages(messages)
#     for message in messages[1:]:
#         print(message)
#     tic = time.time()
#     if model in ['gpt-4o-0806']:
#         response = llm.call(messages=messages,tools=tools,model=model,max_tokens=16384)
#     elif model in ['DeepSeek-R1-0528', 'qwen3-vl-235b-a22b-thinking']:
#         response = llm.call(messages=messages,tools=tools,model=model,max_tokens=30000)
#     else:
#         response = llm.call(messages=messages,tools=tools,model=model)
#     toc = time.time()
#     print(response)
#     success = False
#     if response.get('tool_calls',None):
#         tool_calls = response.get('tool_calls',None)
#         if tool_calls:
#             formatted_tool_calls = []
#             for tc in tool_calls:
#                 # 转换格式：{id, name, input} -> {id, type, function: {name, arguments}}
#                 import json as _json
#                 formatted_tc = {
#                     "id": tc.get("id", ""),
#                     "type": "function",
#                     "function": {
#                         "name": tc.get("name", ""),
#                         "arguments": _json.dumps(tc.get("input", {}), ensure_ascii=False)
#                     }
#                 }
#                 formatted_tool_calls.append(formatted_tc)

#         message  = AssistantMessage(role='assistant',content=None,tool_calls=formatted_tool_calls)
#         success = True
#         tokens_info = response['raw_response']
#         message.usage = tokens_info
#         message.cost = toc-tic
#         return message,success
#     elif response.get('content',None):
#         content = response.get('content',None)
#         message  = AssistantMessage(role='assistant',content=content)
#         success = True
#         tokens_info = response['raw_response']
#         message.usage = tokens_info
#         message.cost = toc-tic
#         return message,success
#     else:
#         return None,success





def generate(    
    model: str,
    messages: list[Message],
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None,
    **kwargs:Any):
    pass
    # timeout = 60
    # max_retries  = 3

    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    messages = to_llm_messages(messages)
    # for message in messages[1:]:
    #     print(message)
    tic = time.time()

    response = call_llm(messages = messages, tools = tools, model = model,**kwargs)

    toc = time.time()
    #print(response)
    success = False
    if response.tool_calls:
        tool_calls = response.tool_calls
        if tool_calls:
            formatted_tool_calls = []
            for tc in tool_calls:
                # 转换格式：{id, name, input} -> {id, type, function: {name, arguments}}
                import json as _json
                formatted_tc = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json.dumps(tc.input, ensure_ascii=False)
                    }
                }
                formatted_tool_calls.append(formatted_tc)

        message  = AssistantMessage(role='assistant',content=None,tool_calls=formatted_tool_calls)
        success = True
        tokens_info = response['raw_response']
        message.usage = tokens_info
        message.cost = toc-tic
        return message,success
    elif response.content:
        content = response.content
        message  = AssistantMessage(role='assistant',content=content)
        success = True
        tokens_info = response.usage
        if tokens_info is not None:
            from dataclasses import asdict
            message.usage = asdict(tokens_info)
        else:
            message.usage = None
        message.cost = toc-tic
        return message,success
    else:
        return None,success



def main():
    pass
    

if __name__ == "__main__":
    main()