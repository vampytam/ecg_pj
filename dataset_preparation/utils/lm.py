from pprint import pprint
from openai import OpenAI

def get_cur_client():
    if os.environ.get('CLIENT_TYPE') == "openai":
        client = OpenAI(
            base_url=os.environ.get('OPENAI_BASE_URL'),
            api_key=os.environ.get('OPENAI_API_KEY'),
        )
        return client
    return None

def get_lm_response():
    client = get_cur_client()
    if client is None:
        return None
    
    response = client.chat.completions.create(
        model=os.environ.get('CLIENT_MODEL'),
        messages=[{
            'role':
                'user',
            'content': [{
                'type': 'text',
                'text': '描述这幅图',
            }, {
                'type': 'image_url',
                'image_url': {
                    'url':
                        'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg',
                },
            }],
        }],
        stream=True
    )
    
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




if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()   
    # response = get_lm_response()
    
    # done_reasoning = False
    # for chunk in response:
    #     reasoning_chunk = chunk.choices[0].delta.reasoning_content
    #     answer_chunk = chunk.choices[0].delta.content
    #     if reasoning_chunk != '':
    #         print(reasoning_chunk, end='',flush=True)
    #     elif answer_chunk != '':
    #         if not done_reasoning:
    #             print('\n\n === Final Answer ===\n')
    #             done_reasoning = True
    #         print(answer_chunk, end='',flush=True)
    
    reasoning_str, answer_str = get_lm_response()
    print("=== Reasoning ===")
    print(reasoning_str)
    print("=== Final Answer ===")
    print(answer_str)