import os
import random
from PIL import Image
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor

# ========== 基础配置 ==========
DATASET_ROOT = "./ecg_class"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# 可配置的 system prompt（也可通过环境变量 ECG_SYSTEM_PROMPT 覆盖）
SYSTEM_PROMPT = os.environ.get(
    "ECG_SYSTEM_PROMPT",
    "You are an expert cardiology assistant specializing in ECG interpretation. "
    "Classify ECG images accurately and be concise."
)

# ========== 1. 读取类别 ==========
def get_class_names(root_dir: str) -> List[str]:
    class_names = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    if not class_names:
        raise ValueError(f"No class subdirectories found under {root_dir}")
    print(f"[INFO] Found {len(class_names)} classes:", class_names)
    return class_names


# ========== 2. 数据划分函数 ==========
def split_dataset(root_dir: str, split_ratio=(0.6, 0.2, 0.2)):
    class_names = get_class_names(root_dir)
    data_splits = {"train": [], "val": [], "test": []}

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)
        images = [os.path.join(class_dir, f)
                  for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        images.sort()

        n_total = len(images)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        random.shuffle(images)
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, img_list in splits.items():
            for img_path in img_list:
                data_splits[split_name].append({
                    "image": img_path,
                    "label": label_idx,
                    "label_text": class_name
                })

    for k, v in data_splits.items():
        print(f"[INFO] {k}: {len(v)} samples")
    return data_splits, class_names


# ========== 3. Prompt 构造函数 ==========
def build_prompt(question: str, class_names: List[str]) -> str:
    classes_str = ", ".join(class_names)
    return (
        f"{question}\n"
        f"The possible classes are: [{classes_str}].\n"
        # f"You should think step by step, output the thinking process in <think></think>\n"
        # f"And then choose one of the most possible class matched to the ECG image, provide that class in <answer></answer> tags."
        f"Please choose one of the most possible class matched to the ECG image.\n"
        f"The answer should be enclosed within <answer> ... </answer> tags"
        # f"if you dont know, please directly respond: 'I dont know'" 
    )


# ========== 4. Dataset 类 ==========
class VLMClassificationDataset(Dataset):
    def __init__(self, samples: List[Dict], class_names: List[str]):
        self.samples = samples
        self.class_names = class_names
        self.question = "Classify the ECG image into one of the given diagnostic categories."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = build_prompt(self.question, self.class_names)
        return {
            "image": item["image"],
            "image_path": item["image"],  # Add image path for wandb logging
            "label": item["label"],
            "label_text": item["label_text"],
            "prompt": prompt
        }


# ========== 5. collate_fn ==========
def collate_fn(batch):
    images, labels, label_texts, prompts, image_paths = [], [], [], [], []

    for item in batch:
        images.append(Image.open(item["image"]).convert("RGB"))
        image_paths.append(item.get("image_path", item["image"]))  # Add image paths
        labels.append(item["label"])
        label_texts.append(item["label_text"])
        prompts.append(item['prompt'])

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Return raw data; processing is done in the training loop
    return {
        "images": images,              # list of PIL.Image
        "image_paths": image_paths,    # list of image paths for wandb logging
        "prompts": prompts,            # list of prompt strings (includes <|image|> token)
        "labels": labels_tensor,       # tensor of label indices
        "label_texts": label_texts,    # list of label strings
    }


# ========== 6. 构建 DataLoader ==========
def build_dataloaders(root_dir, batch_size=4, num_workers=4):
    data_splits, class_names = split_dataset(root_dir)
    datasets = {
        k: VLMClassificationDataset(v, class_names)
        for k, v in data_splits.items()
    }

    loaders = {
        k: DataLoader(
            v,
            batch_size=batch_size,
            shuffle=(k == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        for k, v in datasets.items()
    }

    return loaders, class_names


# ========== 7. TRL 兼容的数据集 ==========
class VLMTRLClassificationDataset(Dataset):
    def __init__(self, samples: List[Dict], class_names: List[str]):
        self.samples = samples
        self.class_names = class_names
        self.question = "Classify the ECG image into one of the given diagnostic categories."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = build_prompt(self.question, self.class_names)
        image = Image.open(item["image"]).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return {
            "messages": messages,
            "prompt": messages,
            "label_text": item["label_text"],
            "image_path": item["image"],  # Add image path for wandb logging
        }


def build_trl_datasets(root_dir):
    data_splits, class_names = split_dataset(root_dir)
    datasets = {
        k: VLMTRLClassificationDataset(v, class_names)
        for k, v in data_splits.items()
    }
    return datasets, class_names


# ========== 7. 调试运行 ==========
if __name__ == "__main__":
    loaders, class_names = build_dataloaders(DATASET_ROOT)
    batch = next(iter(loaders["train"]))

    print("Batch keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Pixel values shape:", batch["pixel_values"].shape)
    print("Labels:", batch["labels"])
    print("Label texts:", batch["label_texts"])