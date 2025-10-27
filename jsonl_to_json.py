import json

def simple_jsonl_to_json(jsonl_file, output_file):
    result = {}
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if isinstance(data, dict):
                    result.update(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

simple_jsonl_to_json("litfl_raw.jsonl", "litfl_llm_refined.json")