# ft_qwen2vl2b_grpo_with_accuracy.py

import os
import torch
import re
from typing import Optional, List, Dict
from datasets import load_dataset
from transformers import AutoProcessor, AutoConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

def format_reward(completions: List[str], **kwargs) -> List[float]:
    """Reward function that checks format: <think>…</think><answer>…</answer>."""
    pattern = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    rewards = [1.0 if re.match(pattern, content, re.DOTALL) else 0.0
               for content in completions]
    return rewards

def accuracy_reward(
    completions: List[str],
    solution: List[str],
    **kwargs
) -> List[Optional[float]]:
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text."""
    rewards: List[Optional[float]] = []
    for completion, sol in zip(completions, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception:
            gold_parsed = []
        if len(gold_parsed) != 0:
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # Fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())
        rewards.append(reward)
    return rewards

def make_conversation(example: Dict, processor, SYSTEM_PROMPT: str) -> Dict:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": example["problem"]}
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "image": example["image"],
        "solution": example["solution"],
    }

def main():
    # 1. Authenticate

    # 2. Load dataset
    dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
    full = load_dataset(dataset_id, split='train[:5%]')
    split = full.train_test_split(test_size=0.2, seed=42)
    train_dataset = split['train']
    test_dataset = split['test']

    # 3. Processor & config
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    config = AutoConfig.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
    SYSTEM_PROMPT = (
        "A conversation between a User and an Assistant. The assistant first thinks about the reasoning process, enclosed in <think>…</think>, "
        "then gives the answer within <answer>…</answer>."
    )

    # 4. Prepare dataset
    train_dataset = train_dataset.map(lambda ex: make_conversation(ex, processor, SYSTEM_PROMPT))
    train_dataset = train_dataset.remove_columns(['problem', 'original_question', 'original_answer'])

    # 5. Load model + LoRA
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Training args with GRPO
    training_args = GRPOConfig(
        output_dir="Qwen2-VL-2B-Instruct-GRPO",
        learning_rate=1e-5,
        remove_unused_columns=False,
        num_train_epochs=1,
        bf16=True,
        per_device_train_batch_size=2,
        max_completion_length=1024,
        num_generations=2,
        max_prompt_length=2048,
        report_to=[],
        logging_steps=10,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=10,
    )

    # 7. Initialize trainer with both reward functions
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    # 8. Qualitative evaluation
    trained_model = Qwen2VLForConditionalGeneration.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    trained_processor = AutoProcessor.from_pretrained(training_args.output_dir, use_fast=True, padding_side="left")

    def generate_with_reasoning(problem, image):
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": problem}
            ]},
        ]
        prompt = trained_processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = trained_processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(trained_model.device)
        with torch.no_grad():
            output_ids = trained_model.generate(**inputs, max_new_tokens=500)
        generated = trained_processor.decode(output_ids[0], skip_special_tokens=True)
        return generated

    example = test_dataset[0]
    print("Input problem:", example["problem"])
    result = generate_with_reasoning(example["problem"], example["image"])
    print("Model response:", result)

if __name__ == "__main__":
    main()
