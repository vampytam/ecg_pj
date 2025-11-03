import torch
import re
import wandb
from typing import Optional, List
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoConfig, Qwen2VLForConditionalGeneration

from dataset import build_trl_datasets, DATASET_ROOT
import dataset as dataset_module

def _completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if hasattr(completion, 'text'):
        return str(completion.text)
    if hasattr(completion, 'content'):
        content = completion.content
        if isinstance(content, str):
            return content
    if isinstance(completion, dict):
        for key in ['text', 'content', 'generated_text', 'output']:
            if key in completion:
                val = completion[key]
                if isinstance(val, str):
                    return val
        content = completion.get("content", None)
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                    parts.append(c["text"])
            return "\n".join(parts)
        return str(completion)
    if isinstance(completion, list):
        parts = []
        for seg in completion:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict) and seg.get("type") == "text" and isinstance(seg.get("text"), str):
                parts.append(seg["text"])
            elif isinstance(seg, dict) and isinstance(seg.get("content"), list):
                for c in seg["content"]:
                    if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text"), str):
                        parts.append(c["text"])
        return "\n".join(parts)
    return str(completion)

def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    return match.group(1).strip() if match else ""

def format_reward(completions: List, **kwargs) -> List[float]:
    pattern = r"<answer>.*?</answer>"
    responses = [_completion_to_text(c) for c in completions]
    return [1.0 if re.search(pattern, resp, re.DOTALL | re.I) else 0.0 for resp in responses]

def accuracy_reward(
    completions: List,
    label_text: Optional[List[str]] = None,
    **kwargs
) -> List[float]:
    if label_text is None:
        label_text = kwargs.get("label_text", [])
    rewards = []
    for completion, gt in zip(completions, label_text):
        response = _completion_to_text(completion)
        ans = extract_answer(response)
        reward = 1.0 if (gt is not None and gt.lower() in ans.lower()) else 0.0
        rewards.append(reward)
    return rewards

# Removed Custom trainer and custom W&B rollout logger for a simpler setup

def main():
    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    OUTPUT_DIR = "./grpo_vlm_classification_out"
    NUM_EPOCHS = 1
    BATCH_SIZE = 2
    LR = 1e-5

    SYSTEM_PROMPT = (
        "You are an expert cardiology assistant specializing in ECG interpretation. "
        "Provide accurate, concise classifications."
    )

    wandb.init(
        project="grpo-ecg-classification",
        name="qwen2vl-2b-grpo-run",
        config={
            "model": MODEL_NAME,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "lora_r": 8,
            "lora_alpha": 32,
            "num_generations": 2,
        }
    )

    dataset_module.SYSTEM_PROMPT = SYSTEM_PROMPT
    trl_datasets, class_names = build_trl_datasets(DATASET_ROOT)
    train_ds = trl_datasets["train"]
    eval_ds = trl_datasets["val"]

    config = AutoConfig.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True, padding_side="left")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, config=config, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        remove_unused_columns=False,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        per_device_train_batch_size=BATCH_SIZE,
        max_completion_length=1024,
        num_generations=2,
        max_prompt_length=2048,
        report_to=["wandb"],
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    trained_model = Qwen2VLForConditionalGeneration.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    trained_processor = AutoProcessor.from_pretrained(
        training_args.output_dir,
        use_fast=True,
        padding_side="left"
    )

    def generate_classification(messages):
        """Generate classification for given messages."""
        prompt = trained_processor.apply_chat_template(messages, add_generation_prompt=True)

        image = None
        for msg in messages:
            if msg["role"] == "user":
                for content in msg["content"]:
                    if content.get("type") == "image":
                        image = content["image"]
                        break

        inputs = trained_processor(
            text=[prompt],
            images=[image] if image else None,
            padding=True,
            return_tensors="pt",
        ).to(trained_model.device)

        with torch.no_grad():
            output_ids = trained_model.generate(**inputs, max_new_tokens=2048)
        generated = trained_processor.decode(output_ids[0], skip_special_tokens=True)
        return generated
    
    if len(eval_ds) > 0:
        print("\n[Evaluation] Testing on validation sample...")
        example = eval_ds[0]
        print(f"  Image path: {example['image_path']}")
        print(f"  Ground truth: {example['label_text']}")

        result = generate_classification(example["messages"])
        print(f"  Model response: {result[:200]}...")
    
    # Evaluate on the test dataset and compute accuracy
    test_ds = trl_datasets.get("test")
    if test_ds is not None and len(test_ds) > 0:
        print("\n[Test] Evaluating on test dataset...")
        num_correct = 0
        total = len(test_ds)
        for sample in test_ds:
            output_text = generate_classification(sample["messages"])
            pred_answer = extract_answer(output_text)
            gt_label = sample.get("label_text", "")
            if gt_label and pred_answer and gt_label.lower() in pred_answer.lower():
                num_correct += 1
        accuracy = num_correct / total if total > 0 else 0.0
        print(f"[Test] Accuracy: {accuracy:.4f} ({num_correct}/{total})")
    wandb.finish()

if __name__:
    main()
