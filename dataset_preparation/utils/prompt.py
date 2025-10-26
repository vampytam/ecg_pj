"""
Prompt构造工具
根据模板和变长参数生成prompt
"""

from typing import Any, Dict, List, Union
import json
import os


class PromptBuilder:
    """Prompt构造器类"""
    
    def __init__(self, template: str = "", template_file: str = None):
        """
        初始化Prompt构造器
        
        Args:
            template: prompt模板字符串，支持{key}格式的占位符
            template_file: 模板文件路径，如果提供则从文件读取模板
        """
        if template_file:
            self.template = self.load_template_from_file(template_file)
        else:
            self.template = template
    
    def load_template_from_file(self, template_file: str) -> str:
        """
        从文件加载模板
        
        Args:
            template_file: 模板文件路径
            
        Returns:
            str: 从文件读取的模板内容
            
        Raises:
            FileNotFoundError: 当模板文件不存在时
            IOError: 当文件读取失败时
        """
        try:
            # 支持相对路径和绝对路径
            if not os.path.isabs(template_file):
                # 相对路径，相对于当前文件所在目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                template_path = os.path.join(current_dir, '..', 'prompt_templates', template_file)
            else:
                template_path = template_file
            
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"模板文件不存在: {template_file}")
        except IOError as e:
            raise IOError(f"读取模板文件失败: {e}")
    
    def set_template(self, template: str) -> 'PromptBuilder':
        """
        设置prompt模板
        
        Args:
            template: prompt模板字符串
            
        Returns:
            self: 返回自身以支持链式调用
        """
        self.template = template
        return self
    
    def set_template_from_file(self, template_file: str) -> 'PromptBuilder':
        """
        从文件设置prompt模板
        
        Args:
            template_file: 模板文件路径
            
        Returns:
            self: 返回自身以支持链式调用
        """
        self.template = self.load_template_from_file(template_file)
        return self
    
    def build(self, **kwargs) -> str:
        """
        根据模板和参数构建prompt
        
        Args:
            **kwargs: 变长关键字参数，用于填充模板中的占位符
            
        Returns:
            str: 构建完成的prompt字符串
            
        Raises:
            KeyError: 当模板中有未提供的参数时
            ValueError: 当参数格式不正确时
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"模板中缺少参数: {e}")
        except ValueError as e:
            raise ValueError(f"参数格式错误: {e}")
    
    def build_with_dict(self, params: Dict[str, Any]) -> str:
        """
        使用字典参数构建prompt
        
        Args:
            params: 参数字典
            
        Returns:
            str: 构建完成的prompt字符串
        """
        return self.build(**params)
    
    def build_with_list(self, *args) -> str:
        """
        使用位置参数构建prompt（模板使用{0}, {1}等格式）
        
        Args:
            *args: 位置参数列表
            
        Returns:
            str: 构建完成的prompt字符串
        """
        try:
            return self.template.format(*args)
        except (IndexError, ValueError) as e:
            raise ValueError(f"位置参数错误: {e}")
    
    def validate_template(self) -> List[str]:
        """
        验证模板并返回所需的参数名列表
        
        Returns:
            List[str]: 模板中需要的参数名列表
        """
        import re
        # 查找所有{key}格式的占位符
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, self.template)
        return list(set(matches))  # 去重
    
    def get_template_info(self) -> Dict[str, Any]:
        """
        获取模板信息
        
        Returns:
            Dict[str, Any]: 包含模板和所需参数的信息
        """
        required_params = self.validate_template()
        return {
            'template': self.template,
            'required_params': required_params,
            'param_count': len(required_params)
        }


def create_prompt_from_file(template_file: str, **kwargs) -> str:
    """
    从文件创建prompt的便捷函数
    
    Args:
        template_file: 模板文件路径
        **kwargs: 参数
        
    Returns:
        str: 构建完成的prompt
    """
    builder = PromptBuilder(template_file=template_file)
    return builder.build(**kwargs)


def create_prompt_from_file_with_dict(template_file: str, params: Dict[str, Any]) -> str:
    """
    从文件和字典创建prompt的便捷函数
    
    Args:
        template_file: 模板文件路径
        params: 参数字典
        
    Returns:
        str: 构建完成的prompt
    """
    builder = PromptBuilder(template_file=template_file)
    return builder.build_with_dict(params)


def list_template_files() -> List[str]:
    """
    列出prompt_templates目录下的所有模板文件
    
    Returns:
        List[str]: 模板文件名列表
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(current_dir, '..', 'prompt_templates')
    
    if not os.path.exists(template_dir):
        return []
    
    template_files = []
    for file in os.listdir(template_dir):
        if file.endswith('.txt'):
            template_files.append(file)
    
    return sorted(template_files)


def get_template_info_from_file(template_file: str) -> Dict[str, Any]:
    """
    从文件获取模板信息
    
    Args:
        template_file: 模板文件路径
        
    Returns:
        Dict[str, Any]: 包含模板和所需参数的信息
    """
    builder = PromptBuilder(template_file=template_file)
    return builder.get_template_info()


def create_prompt(template: str, **kwargs) -> str:
    """
    快速创建prompt的便捷函数
    
    Args:
        template: prompt模板
        **kwargs: 参数
        
    Returns:
        str: 构建完成的prompt
    """
    builder = PromptBuilder(template)
    return builder.build(**kwargs)


def create_prompt_from_dict(template: str, params: Dict[str, Any]) -> str:
    """
    从字典创建prompt的便捷函数
    
    Args:
        template: prompt模板
        params: 参数字典
        
    Returns:
        str: 构建完成的prompt
    """
    builder = PromptBuilder(template)
    return builder.build_with_dict(params)


# 示例用法
if __name__ == "__main__":
    # 示例1: 基本用法
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
    
    # 示例2: 使用PromptBuilder类
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
    
    # 示例3: 位置参数
    template3 = "分析{0}类型的{1}数据，重点关注{2}方面。"
    builder3 = PromptBuilder(template3)
    prompt3 = builder3.build_with_list("ECG", "心电图", "心律")
    print("示例3:")
    print(prompt3)
    print()
    
    # 示例4: 模板验证
    template4 = "请分析{data_type}数据，关注{aspect1}和{aspect2}，给出{conclusion}。"
    builder4 = PromptBuilder(template4)
    info = builder4.get_template_info()
    print("示例4 - 模板信息:")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print()
    
    # 示例5: 从文件读取模板
    print("示例5 - 从文件读取模板:")
    try:
        # 列出所有可用的模板文件
        template_files = list_template_files()
        print(f"可用的模板文件: {template_files}")
        
        if template_files:
            # 使用第一个模板文件
            template_file = template_files[0]
            print(f"\n使用模板文件: {template_file}")
            
            # 获取模板信息
            template_info = get_template_info_from_file(template_file)
            print("模板信息:")
            print(json.dumps(template_info, ensure_ascii=False, indent=2))
            
            # 使用模板文件创建prompt
            prompt5 = create_prompt_from_file(
                template_file,
                data_type="ECG",
                data_content="心率不规律，QRS波群异常",
                analysis_type="诊断",
                patient_context="中年男性"
            )
            print("\n生成的prompt:")
            print(prompt5)
            
    except Exception as e:
        print(f"从文件读取模板时出错: {e}")
    
    print()
    
    # 示例6: 使用PromptBuilder从文件加载模板
    print("示例6 - 使用PromptBuilder从文件加载模板:")
    try:
        builder6 = PromptBuilder(template_file="patient_consultation.txt")
        prompt6 = builder6.build(
            name="李四",
            age=35,
            symptoms="心悸、头晕",
            advice_type="治疗"
        )
        print(prompt6)
    except Exception as e:
        print(f"使用PromptBuilder从文件加载模板时出错: {e}")
