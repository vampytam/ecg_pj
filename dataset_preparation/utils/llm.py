from pprint import pprint
from openai import OpenAI

import json

def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

config = load_config()

def get_cur_client():
    if config is None:
        return None
    if config["llm"]["client_provider"] == "openai":
        client = OpenAI(
            base_url=config["llm"]["base_url"],
            api_key=config["llm"]["api_key"],
        )
        return client
    return None

def get_lm_response(prompt="", stream=True):
    """
    Get response from LLM model with customizable prompt.
    
    Args:
        prompt (str): Text prompt for the model
        stream (bool): Whether to stream the response
    
    Returns:
        tuple: (reasoning_str, answer_str) or None if error
    """
    client = get_cur_client()
    if client is None:
        return None
    
    response = client.chat.completions.create(
        model=config["llm"]["client_model"],
        messages=[{
            'role': 'user',
            'content': prompt
        }],
        stream=stream
    )
    
    if stream:
        reasoning_str = ''
        answer_str = ''
        for chunk in response:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            answer_chunk = chunk.choices[0].delta.content
            if reasoning_chunk != '':
                reasoning_str += reasoning_chunk
            elif answer_chunk != '':
                answer_str += answer_chunk
                
        return (reasoning_str, answer_str)
    else:
        reasoning_content = response.choices[0].message.reasoning_content
        answer_content = response.choices[0].message.content
        return (reasoning_content or '', answer_content or '')


if __name__ == "__main__":
    # Example usage with different parameters
    
    # Example 1: Basic text prompt
    print("=== Example 1: Basic Text Prompt ===")
    reasoning_str, answer_str = get_lm_response(
        prompt="请解释什么是心电图，并说明其在医学诊断中的重要性。"
    )
    print("=== Reasoning ===")
    print(reasoning_str)
    print("=== Final Answer ===")
    print(answer_str)
    
    # Example 2: Non-streaming response
    # print("\n=== Example 2: Non-streaming ===")
    # reasoning_str, answer_str = get_lm_response(
    #     prompt="分析心电图的各个波形特征",
    #     stream=False
    # )
    # print("=== Reasoning ===")
    # print(reasoning_str)
    # print("=== Final Answer ===")
    # print(answer_str)
    
    # Example 3: Medical diagnosis question
    # print("\n=== Example 3: Medical Diagnosis ===")
    # reasoning_str, answer_str = get_lm_response(
    #     prompt="根据以下心电图特征，可能的诊断是什么：P波正常，QRS波群增宽，T波倒置？"
    # )
    # print("=== Reasoning ===")
    # print(reasoning_str)
    # print("=== Final Answer ===")
    # print(answer_str)
