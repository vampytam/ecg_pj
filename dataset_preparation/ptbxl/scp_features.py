import pandas as pd
import json
import re
from itertools import islice

from ..utils.prompt import create_prompt_from_file
from ..utils.llm import get_lm_response

def extract_columns_from_csv(file_path):
    data = pd.read_csv(file_path)
    
    index_column = data.iloc[:, 0]
    description_column = data['description']
    
    scp_desc_dict = dict(zip(index_column, description_column))
    
    with open('scp_desc.json', 'w', encoding='utf-8') as json_file:
        json.dump(scp_desc_dict, json_file, ensure_ascii=False, indent=4)
    
    return scp_desc_dict


#  k items as a groups
def extract_scp_features(scp_desc_dict, k):
    scp_features = {}
    iter_items = iter(scp_desc_dict.items())
    
    while True:
        batch = list(islice(iter_items, k))
        if not batch:
            break
        
        diagnosis_array = [scp_desc for scp_code, scp_desc in batch]
        prompt = create_prompt_from_file(
            './dataset_preparation/prompt_templates/ecg_features_of_diagnosis.txt',
            diagnosis_array=diagnosis_array
        )
        _, answer_str = get_lm_response(prompt, stream=False)
        
        pattern = re.compile(r'"(.*?)"', flags=re.S)
        ecg_features = [m.group(1) for m in pattern.finditer(answer_str)]
        
        if len(ecg_features) != len(batch):
            print(f"Warning: Mismatch in number of features extracted of batch {diagnosis_array}")
            continue
        
        for (scp_code, scp_desc), ecg_feature in zip(batch, ecg_features):
            scp_features[scp_code] = {
                'description': scp_desc,
                'ecg_features': ecg_feature
            }
            
    with open('scp_features.json', 'w', encoding='utf-8') as json_file:
        json.dump(scp_features, json_file, ensure_ascii=False, indent=4)
    
    return scp_features

def prompt_gen():
    with open('scp_desc.json', 'r', encoding='utf-8') as json_file:
        scp_desc_dict = json.load(json_file)
    
    for val in scp_desc_dict.values():
        diagnosis_array = [val]
        prompt = create_prompt_from_file(
            './dataset_preparation/prompt_templates/ecg_features_of_diagnosis.txt',
            diagnosis_array=diagnosis_array
        )
        # save prompts to a text file
        with open('scp_feature_prompts.txt', 'a', encoding='utf-8') as f:
            f.write(prompt + "\n\n---\n\n")

if __name__ == "__main__":
    # prompt_gen()
    # file_path = '/Users/bacmive/Data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv'
    # scp_desc_dict = extract_columns_from_csv(file_path)
    
    with open('scp_desc.json', 'r', encoding='utf-8') as json_file:
        scp_desc_dict = json.load(json_file)
        
    scp_features= extract_scp_features(scp_desc_dict, k=4)
    
    print("Dictionary saved to scp_desc.json:")
    print(scp_desc_dict)
    print("Extracted SCP features saved to scp_features.json:")
    print(scp_features)
    