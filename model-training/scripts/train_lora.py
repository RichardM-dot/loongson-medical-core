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
# 脚本设计为"在基座模型目录里运行"：MODEL_PATH 指向当前目录，
# 即 models/base/ 下的 Qwen2.5-0.5B-Instruct（config/tokenizer/权重齐全）
MODEL_PATH = "."  # 当前目录有模型文件
OUTPUT_DIR = "lora_adapter"   # 训练完只保存 LoRA 适配器（几十 MB），不存基座
DATA_DIR = "data"             # preprocess_data.py 的输出目录

# LoRA 配置
# r=8：低秩矩阵秩，0.5B 小模型取小 r 即可；alpha=16 → 缩放系数 alpha/r=2
# 7 个 target_modules 覆盖注意力(q/k/v/o)与 MLP(gate/up/down)，即全线性层注入
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,   # 轻度 dropout 抑制小数据过拟合
    bias="none",         # 不训练偏置，只训低秩旁路
    task_type="CAUSAL_LM",
)

# 训练参数
# 总体思路：CPU/低端设备可跑 —— 真实 batch=1，靠梯度累积凑等效 16
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,   # CPU用小batch
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,   # 等效batch_size=16
    warmup_steps=50,                  # 前 50 步线性升温，避免初期学习率过猛
    logging_steps=10,
    save_steps=200,
    eval_steps=200,                   # 每 200 步跑一次验证，与保存对齐
    save_total_limit=2,               # 磁盘只留最近 2 个 checkpoint
    learning_rate=2e-4,               # LoRA 常用 lr，比全参微调(~2e-5)大一个量级
    weight_decay=0.01,
    fp16=False,                       # CPU 训练关闭半精度（CPU 无 TensorCore 收益）
    bf16=False,
    max_grad_norm=0.3,                # 梯度裁剪阈值，LoRA 推荐值，防小数据梯度爆炸
    dataloader_num_workers=0,         # 0 = 主进程加载，省内存（嵌入式/小内存机友好）
    report_to="none",                 # 不上报 wandb/tensorboard，纯本地
    save_strategy="steps",
    eval_strategy="steps",
    load_best_model_at_end=True,      # 训完回载 eval_loss 最优的 checkpoint 而非最后一轮
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,      # 配合 formatting_func：保留 text 列自己拼 prompt
)

# 使用多少训练数据
# 0.5B + CPU 场景的采样上限：1000 条 × 3 epoch = 3000 步内可收敛出可用风格
MAX_TRAIN_SAMPLES = 1000
MAX_VAL_SAMPLES = 200
MAX_SEQ_LENGTH = 512   # 与 preprocess_data.py 的长度过滤(问≤500/答≤1000字)呼应


# ── 加载 JSON 对话数据并转成 HF Dataset ──
def load_data(split: str, max_samples: int) -> Dataset:
    """加载JSON数据并转换为Dataset"""
    path = os.path.join(DATA_DIR, f"{split}.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples and len(data) > max_samples:
        data = data[:max_samples]   # 只取前 N 条（数据已 shuffle 过，等价随机采样）

    # 转换为文本格式 (Qwen Chat模板)
    texts = []
    for item in data:
        messages = item["messages"]
        # 手动构建chat文本：不用 tokenizer.apply_chat_template，
        # 手拼 ChatML 保证与 evaluate.py / assistant.c 的推理格式完全一致
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        texts.append("\n".join(text_parts))

    # 只留一列 text，后续由 SFTTrainer + formatting_func 消费
    return Dataset.from_dict({"text": texts})


# ── 训练主流程：载模型 -> 注入 LoRA -> SFT -> 只保存适配器 ──
def main():
    print("=" * 50)
    print("加载模型和分词器...")
    device = "cuda" if torch.cuda.is_available() else "cpu"   # 有 GPU 用 GPU，否则回落 CPU
    print(f"设备: {device}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,   # 允许执行模型仓库自带的自定义代码（Qwen 需要）
    )

    # 确保有pad_token
    # Qwen 的 tokenizer 没配 pad：批量训练时用 eos 填充，
    # SFTTrainer 内部会把 pad 位置的 label 置 -100 不计损失
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"分词器加载完成, vocab_size={tokenizer.vocab_size}")

    # 加载模型 (全精度, CPU friendly)
    print("加载模型 (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,   # 0.5B fp32 ≈ 2GB 内存，小模型全精度训练可行
        device_map={"": device},     # 全部层放同一设备
        trust_remote_code=True,
        low_cpu_mem_usage=True,      # 分块加载权重，峰值内存近乎减半
    )

    print(f"模型加载完成, 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # 加载数据
    print("加载训练数据...")
    train_dataset = load_data("train", MAX_TRAIN_SAMPLES)
    eval_dataset = load_data("val", MAX_VAL_SAMPLES)
    print(f"训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")

    # 创建训练器 (trl 1.7+ API, 自动处理LoRA)
    def formatting_func(example):
        return example["text"]   # 取 load_data 拼好的整段 ChatML 文本

    # peft_config 传入后：SFTTrainer 自动把基座包成 PeftModel，
    # 冻结全部基座权重、只训 LoRA 旁路
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
        trainer.model.print_trainable_parameters()   # 预期 <1% 参数量可训

    print("开始训练...")
    trainer.train()   # 结束后自动 load_best_model_at_end 回载最优权重

    # 保存LoRA适配器
    print("保存LoRA适配器...")
    trainer.model.save_pretrained(OUTPUT_DIR)   # 只存 A/B 低秩矩阵，不存基座
    tokenizer.save_pretrained(OUTPUT_DIR)       # 分词器随行，便于合并脚本直接用

    print(f"训练完成! 适配器保存在: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
