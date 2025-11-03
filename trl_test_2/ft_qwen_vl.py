# script: fine_tune_vlm_trl.py

import os
import torch
import gc
import time

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from qwen_vl_utils import process_vision_info

def clear_memory():
    """Free GPU memory."""
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def format_data(sample, system_message: str):
    """Format a single dataset sample into the chat-style dict expected by the processor."""
    return {
        "images": [sample["image"]],
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"]
                    },
                    {
                        "type": "text",
                        "text": sample["query"]
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["label"][0]
                    }
                ],
            },
        ]
    }

def generate_text_from_sample(model, processor, sample, device="cuda", max_new_tokens=1024):
    """Helper to run inference on one sample."""
    # prepare text input (skip system message when evaluating)
    text_input = processor.apply_chat_template(
        sample["messages"][1:2],  # user only
        tokenize=False,
        add_generation_prompt=True
    )

    # process vision info
    image_inputs, _ = process_vision_info(sample["messages"])

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt"
    ).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]

def main():
    # 1. Install dependencies
    # (assumed done outside of script via pip install)
    # pip install -U git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio
    # tested with trl==0.22.0.dev0, bitsandbytes==0.47.0, peft==0.17.1, qwen-vl-utils==0.0.11, trackio==0.2.8 :contentReference[oaicite:3]{index=3}

    # 2. Load dataset
    dataset_id = "HuggingFaceM4/ChartQA"  # change to your dataset
    train_ds, eval_ds, test_ds = load_dataset(dataset_id, split=["train[:10%]", "val[:10%]", "test[:10%]"])
    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colours, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
    train_dataset = [ format_data(sample, system_message) for sample in train_ds ]
    eval_dataset  = [ format_data(sample, system_message) for sample in eval_ds ]
    test_dataset  = [ format_data(sample, system_message) for sample in test_ds ]

    # 3. Load model & processor and test baseline
    model_id = "Qwen/Qwen2-VL-2B-Instruct"  # change if you use another model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    # Example test
    print("Example baseline output:")
    print(generate_text_from_sample(model, processor, train_dataset[0]))

    # 4. Fine-tune the model using TRL
    # 4.1 Load model with quantization for training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    # 4.2 Set up QLoRA/LoRA config + SFTConfig
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir="qwen2-7b-instruct-trl-sft-ChartQA",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model = peft_model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        peft_config = peft_config,
        processing_class = processor
    )

    # Train
    # trainer.train()

    # Save model
    # trainer.save_model(training_args.output_dir)

    # 5. Testing the fine-tuned model
    clear_memory()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)

    adapter_path = training_args.output_dir  # or your hub path
    peft_model = get_peft_model(model, peft_config)
    peft_model.load_adapter(adapter_path, adapter_name="default")
    peft_model.set_adapter("default")

    print("Example after fine-tuning (on train sample):")
    print(generate_text_from_sample(peft_model, processor, train_dataset[0]))

    print("Example after fine-tuning (on unseen test sample):")
    print(generate_text_from_sample(peft_model, processor, test_dataset[10]))

if __name__ == "__main__":
    main()
