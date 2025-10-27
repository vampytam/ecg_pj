import pandas as pd
import json

def extract_columns_from_csv(file_path):
    data = pd.read_csv(file_path)
    
    index_column = data.iloc[:, 0]
    description_column = data['description']
    
    scp_desc_dict = dict(zip(index_column, description_column))
    
    with open('scp_desc.json', 'w', encoding='utf-8') as json_file:
        json.dump(scp_desc_dict, json_file, ensure_ascii=False, indent=4)
    
    return scp_desc_dict


def extract_scp_features(scp_desc_dict):
    scp_features = []
    for scp_code, scp_desc in scp_desc_dict.items():
        scp_features.append({
            "scp_code": scp_code,
            "scp_desc": scp_desc
        })
    
    return scp_features

if __name__ == "__main__":
    file_path = '/Users/bacmive/Data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv'
    scp_desc_dict = extract_columns_from_csv(file_path)
    
    print("Dictionary saved to scp_desc.json:")
    print(scp_desc_dict)