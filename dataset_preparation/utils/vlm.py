from pprint import pprint
from openai import OpenAI

import json

class VLMClient:
    _instance = None
    _config = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLMClient, cls).__new__(cls)
            cls._config = cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Error: config.json not found")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in config.json")
            return None

    def get_client(self):
        if self._config is None:
            return None
        if self._client is None or self._config["vlm"]["client_provider"] != "openai":
            self._client = OpenAI(
                base_url=self._config["vlm"]["base_url"],
                api_key=self._config["vlm"]["api_key"],
            )
        return self._client

def get_lm_response(prompt="", image=None, image_url=None, stream=True):
    client = VLMClient().get_client()
    if client is None:
        return None
    
    content = [{'type': 'text', 'text': prompt}]
    
    if image_url:
        content.append({
            'type': 'image_url',
            'image_url': {'url': image_url}
        })
    elif image:
        import base64
        import mimetypes        
        mime_type, _ = mimetypes.guess_type(image)
        if not mime_type or not mime_type.startswith('image/'):
            print(f"Error: {image} is not a valid image file")
            return None        
        try:
            with open(image, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:{mime_type};base64,{img_data}"
                    }
                })
        except FileNotFoundError:
            print(f"Error: Image file {image} not found")
            return None
        except Exception as e:
            print(f"Error reading image {image}: {e}")
            return None
    
    response = client.chat.completions.create(
        model=VLMClient._config["vlm"]["client_model"],
        messages=[{
            'role': 'user',
            'content': content
        }],
        stream=stream
    )
    
    if stream:
        reasoning_str = ''
        answer_str = ''
        for chunk in response:
            reasoning_chunk = getattr(chunk.choices[0].delta, "reasoning_content", "")
            answer_chunk = chunk.choices[0].delta.content
            if reasoning_chunk:
                reasoning_str += reasoning_chunk
            elif answer_chunk:
                answer_str += answer_chunk
                
        return (reasoning_str, answer_str)
    else:
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", "")
        answer_content = response.choices[0].message.content
        return (reasoning_content, answer_content)




if __name__ == "__main__":
    # Example usage with different parameters
    
    # Example 1: Using image URL (default behavior)
    # print("=== Example 1: Image URL ===")
    # reasoning_str, answer_str = get_lm_response(
    #     prompt="详细描述这张图片中的内容",
    #     image_url="https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg"
    # )
    # print("=== Reasoning ===")
    # print(reasoning_str)
    # print("=== Final Answer ===")
    # print(answer_str)
    
    # Example 2: Using local image file
    print("\n=== Example 2: Local Image ===")
    reasoning_str, answer_str = get_lm_response(
        prompt="Interpret the provided ECG image, identify key features and abnormalities in each lead, and generate a clinical diagnosis that is supported by the observed evidence.",
        image="/Users/bacmive/Codes/python/ecg/ecg-image-kit/codes/ecg-image-generator/output/41328635-0.png"
    )
    print("=== Reasoning ===")
    print(reasoning_str)
    print("=== Final Answer ===")
    print(answer_str)
    
    # Example 3: Text-only prompt (no image)
    # print("\n=== Example 3: Text Only ===")
    # reasoning_str, answer_str = get_lm_response(
    #     prompt="请解释什么是心电图",
    #     stream=False
    # )
    # print("=== Reasoning ===")
    # print(reasoning_str)
    # print("=== Final Answer ===")
    # print(answer_str)