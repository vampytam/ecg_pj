from typing import Any, Dict, List, Union
import json
import os


class PromptBuilder:
    def __init__(self, template: str = "", template_file: str = None):
        if template_file:
            self.template = self.load_template_from_file(template_file)
        else:
            self.template = template
    
    def load_template_from_file(self, template_file: str) -> str:
        try:            
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise Exception(f"when load template from file:{e}")
    
    def set_template(self, template: str) -> 'PromptBuilder':
        self.template = template
        return self
    
    def set_template_from_file(self, template_file: str) -> 'PromptBuilder':
        self.template = self.load_template_from_file(template_file)
        return self
    
    def build(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"missing required param: {e}")
        except ValueError as e:
            raise ValueError(f"invalid param: {e}")
    
    def build_with_dict(self, params: Dict[str, Any]) -> str:
        return self.build(**params)
    
    def build_with_list(self, *args) -> str:
        try:
            return self.template.format(*args)
        except (IndexError, ValueError) as e:
            raise ValueError(f"wehen build with list, positional params error: {e}")
    
    def validate_template(self) -> List[str]:
        import re
        # 查找所有{key}格式的占位符
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, self.template)
        return list(set(matches))
    
    def get_template_info(self) -> Dict[str, Any]:
        required_params = self.validate_template()
        return {
            'template': self.template,
            'required_params': required_params,
            'param_count': len(required_params)
        }


def create_prompt_from_file(template_file: str, **kwargs) -> str:
    builder = PromptBuilder(template_file=template_file)
    return builder.build(**kwargs)


def create_prompt_from_file_with_dict(template_file: str, params: Dict[str, Any]) -> str:
    builder = PromptBuilder(template_file=template_file)
    return builder.build_with_dict(params)


def list_template_files(template_dir: str = None) -> List[str]:
    if not os.path.exists(template_dir):
        return []
    
    template_files = []
    for file in os.listdir(template_dir):
        if file.endswith('.txt'):
            template_files.append(file)
    
    return sorted(template_files)


def get_template_info_from_file(template_file: str) -> Dict[str, Any]:
    builder = PromptBuilder(template_file=template_file)
    return builder.get_template_info()


def create_prompt(template: str, **kwargs) -> str:
    builder = PromptBuilder(template)
    return builder.build(**kwargs)


def create_prompt_from_dict(template: str, params: Dict[str, Any]) -> str:
    builder = PromptBuilder(template)
    return builder.build_with_dict(params)


if __name__ == "__main__":
    # example 1
    template1 = "请分析以下{data_type}数据：{data_content}，并给出{analysis_type}建议。"
    prompt1 = create_prompt(
        template1,
        data_type="ECG",
        data_content="心率不规律，QRS波群异常",
        analysis_type="诊断"
    )
    print("示例1:")
    print(prompt1)
    print()
    
    # example 2
    builder = PromptBuilder()
    builder.set_template("患者{name}，年龄{age}岁，症状：{symptoms}。请提供{advice_type}建议。")
    
    patient_info = {
        "name": "张三",
        "age": 45,
        "symptoms": "胸痛、气短",
        "advice_type": "治疗"
    }
    
    prompt2 = builder.build_with_dict(patient_info)
    print("示例2:")
    print(prompt2)
    print()
    
    # example 3
    template3 = "分析{0}类型的{1}数据，重点关注{2}方面。"
    builder3 = PromptBuilder(template3)
    prompt3 = builder3.build_with_list("ECG", "心电图", "心律")
    print("示例3:")
    print(prompt3)
    print()
    
    # example 4
    template4 = "请分析{data_type}数据，关注{aspect1}和{aspect2}，给出{conclusion}。"
    builder4 = PromptBuilder(template4)
    info = builder4.get_template_info()
    print("示例4 - 模板信息:")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print()
    
    # example 5
    print("示例5 - 从文件读取模板:")
    try:
        template_dir="./dataset_preparation/prompt_templates"
        template_files = list_template_files(template_dir=template_dir)
        print(f"可用的模板文件: {template_files}")
        
        if template_files:
            template_file = template_files[0]
            print(f"\n使用模板文件: {template_file}")
            
            template_info = get_template_info_from_file(os.path.join(template_dir, template_file))
            print("模板信息:")
            print(json.dumps(template_info, ensure_ascii=False, indent=2))
            
            prompt5 = create_prompt_from_file(
                os.path.join(template_dir, template_file),
                title="心律失常检测",
                diag_info="心率不规律，QRS波群异常",
            )
            print("\n生成的prompt:")
            print(prompt5)
            
    except Exception as e:
        print(f"从文件读取模板时出错: {e}")
    
    print()
    
    # example 6
    print("示例6 - 使用PromptBuilder从文件加载模板:")
    try:
        builder6 = PromptBuilder(template_file=os.path.join(template_dir,  "litfl_refine.txt"))
        prompt6 = builder6.build(
            name="李四",
            title="心悸、头晕",
            diag_info="治疗"
        )
        print(prompt6)
    except Exception as e:
        print(f"使用PromptBuilder从文件加载模板时出错: {e}")
