#!/usr/bin/env python3
"""LoRA微调 Qwen2.5-0.5B-Instruct 用于医疗问诊 (CPU全精度+LoRA)"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

# ============ 配置 ============
MODEL_PATH = "."  # 当前目录有模型文件
OUTPUT_DIR = "lora_adapter"
DATA_DIR = "data"

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,   # CPU用小batch
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,   # 等效batch_size=16
    warmup_steps=50,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    save_total_limit=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    dataloader_num_workers=0,
    report_to="none",
    save_strategy="steps",
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
)

# 使用多少训练数据
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 200
MAX_SEQ_LENGTH = 512


def load_data(split: str, max_samples: int) -> Dataset:
    """加载JSON数据并转换为Dataset"""
    path = os.path.join(DATA_DIR, f"{split}.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    
    # 转换为文本格式 (Qwen Chat模板)
    texts = []
    for item in data:
        messages = item["messages"]
        # 手动构建chat文本
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        texts.append("\n".join(text_parts))
    
    return Dataset.from_dict({"text": texts})


def main():
    print("=" * 50)
    print("加载模型和分词器...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    
    # 确保有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"分词器加载完成, vocab_size={tokenizer.vocab_size}")
    
    # 加载模型 (全精度, CPU friendly)
    print("加载模型 (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    print(f"模型加载完成, 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    
    # 加载数据
    print("加载训练数据...")
    train_dataset = load_data("train", MAX_TRAIN_SAMPLES)
    eval_dataset = load_data("val", MAX_VAL_SAMPLES)
    print(f"训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")
    
    # 创建训练器 (trl 1.7+ API, 自动处理LoRA)
    def formatting_func(example):
        return example["text"]
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        peft_config=lora_config,
    )
    
    print("LoRA可训练参数:")
    if hasattr(trainer.model, 'print_trainable_parameters'):
        trainer.model.print_trainable_parameters()
    
    print("开始训练...")
    trainer.train()
    
    # 保存LoRA适配器
    print("保存LoRA适配器...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"训练完成! 适配器保存在: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
