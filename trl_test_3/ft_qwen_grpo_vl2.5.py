# vlm_grpo_trl_training.py

import os
import torch
import re
from datasets import load_dataset
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from trl import GRPOConfig, GRPOTrainer
from qwen_vl_utils import process_vision_info
from huggingface_hub import login

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the correct <think> … </think><answer> … </answer> format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def accuracy_reward(completions: list[list[dict[str,str]]], solution: list[str], **kwargs):
    """Reward function that checks correctness of the solution compared to the gold-solution."""
    rewards = []
    for completion, sol in zip(completions, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []
        if len(gold_parsed) != 0:
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config = [
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
            reward = float(completion.strip().lower() == sol.strip().lower())
        rewards.append(reward)
    return rewards

def make_conversation(example, processor, SYSTEM_PROMPT):
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
    # 1. Install / authenticate (outside Python or via pip)
    #   !pip install -U git+https://github.com/huggingface/trl.git peft math_verify qwen-vl-utils[decord]

    # 2. Load dataset
    dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
    full = load_dataset(dataset_id, split='train[:5%]')
    split = full.train_test_split(test_size=0.2, seed=42)
    train_dataset = split['train']
    test_dataset = split['test']

    # 2b. Processor & prompt setup
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    # 2c. Convert dataset for training
    train_dataset = train_dataset.map(lambda ex: make_conversation(ex, processor, SYSTEM_PROMPT))
    train_dataset = train_dataset.remove_columns(['problem', 'original_question', 'original_answer'])
    # we assume solution column remains for reward function

    # 3. Load model and apply LoRA
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_id,
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

    # 4. Configure GRPO training parameters
    training_args = GRPOConfig(
        output_dir="Qwen2.5-VL-3B-Instruct-Thinking",
        learning_rate=1e-5,
        remove_unused_columns=False,
        num_train_epochs=1,
        bf16=True,
        per_device_train_batch_size=2,
        max_completion_length=1024,
        num_generations=2,
        max_prompt_length=2048,
        report_to="none",
        logging_steps=10,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=10,
    )

    # 5. Initialize and run trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

    # 6. Evaluate performance qualitatively
    trained_model_id = "your-username/Qwen2.5-VL-3B-Instruct-Thinking"
    trained_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        trained_model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    trained_processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

    def generate_with_reasoning(problem, image):
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": problem}
            ]},
        ]
        prompt = trained_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = trained_processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(trained_model.device)
        with torch.no_grad():
            output_ids = trained_model.generate(**inputs, max_new_tokens=500)
        generated_text = trained_processor.decode(output_ids[0], skip_special_tokens=True)
        # (you can also measure inference time / generated tokens if needed)
        return generated_text

    # Example usage (on test dataset)
    problem = test_dataset[0]['problem']
    image = test_dataset[0]['image']
    print("Input problem:", problem)
    generated, = generate_with_reasoning(problem, image),
    print("Model response:", generated)

if __name__ == "__main__":
    main()
