import time
import json
import requests
import dashscope
import threading

from typing import List, Callable
from queue import Queue
from http import HTTPStatus
from dashscope import Generation
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-Nxkb8NTYp4"
dashscope.api_key = API_KEY

def call_qwen_single(prompt: str, model="qwen-max", max_try: int = 30):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}]
    for _ in range(max_try):
        try:
            response = Generation.call(model=model,
                                    messages=messages,
                                    result_format='message')
            if response.status_code == HTTPStatus.OK:
                if response.output.choices[0]['message']['content'] is not None:
                    return response.output.choices[0]['message']['content']
                else:
                    # print(response)
                    raise RuntimeError("qwen需要重试")
            else:
                # print(response)
                raise RuntimeError("qwen需要重试")
        except Exception as e:
            print(f"qwen 调用失败，重试中: {str(e)}")
            # raise RuntimeError("qwen 没调用成功")
            time.sleep(3)
    
    # import pdb;pdb.set_trace()
    # 可能因为安全问题被过滤了
    return "安全问题被屏蔽" # 

def execute_call_llm_concurrently(
    prompt: List[str],
    model="qwen-max",
    max_try: int = 5,
    # max_workers: int = 3
) -> List[str]:
    """多线程调用qwen-max/gpt"""
    call_llm_single = call_qwen_single if "qwen" in model else call_gpt_single
    max_workers = len(prompt)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_llm_single, prompt[i], model, max_try)
                   for i in range(len(prompt))]
        return [future.result() for future in futures]
