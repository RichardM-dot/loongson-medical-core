#!/usr/bin/env python3
"""评估微调效果: 对比微调前后模型在医疗问诊上的表现"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

MODEL_PATH = "."                 # 基座模型目录（与 train_lora.py 同一约定）
LORA_PATH = "lora_adapter"       # 待评估的 LoRA 适配器
TEST_DATA = "data/test.json"     # preprocess_data.py 划出的测试集
NUM_SAMPLES = 10  # 测试样本数   # 实际按科室抽样：每科室最多 3 条（见 main）


# ── 构建与训练时一致的 ChatML prompt（不含 assistant 回答段）──
def format_prompt(messages: list) -> str:
    """构建Qwen Chat格式prompt"""
    text_parts = []
    for msg in messages:
        text_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    prompt = "\n".join(text_parts)
    # 只保留到assistant开始
    return prompt   # 调用方自行追加 "<|im_start|>assistant\n" 作为生成起点


# ── 单条推理：采样生成，返回模型新增的那段文本 ──
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():   # 评估不需要梯度，省一半内存
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,   # 0.5B CPU 推理：256 token 约几秒~十几秒
            temperature=0.7,      # Qwen2.5 官方推荐采样温度（比默认 1.0 更稳）
            top_p=0.8,
            top_k=20,
            do_sample=True,       # 采样生成而非贪心，更接近真实使用场景
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只提取新生成的部分
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]   # 切掉 prompt 回显，只留续写
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ── 极简 BLEU-1（字符级）：候选字符集与参考字符集的重合率 ──
def simple_bleu(reference: str, candidate: str) -> float:
    """简单BLEU-1计算 (字符级别)"""
    # 注意：这是"字符集合重叠率"，不是标准 BLEU（无 n-gram、无短惩罚），
    # 只能用于同条件下两模型的相对比较，绝对值无语言学意义
    ref_chars = set(reference)
    cand_chars = set(candidate)
    if not cand_chars:
        return 0.0   # 空输出记 0 分，不参与 NaN
    overlap = len(ref_chars & cand_chars)
    return overlap / len(cand_chars)


# ── 评估主流程：抽样 -> 跑基座 -> 跑 LoRA -> 算分对比 -> 出 JSON/MD 报告 ──
def main():
    print("=" * 60)
    print("医疗问诊模型评估")
    print("=" * 60)

    # 加载测试数据
    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 各科室各抽几题
    dept_samples = {}
    for item in test_data[:500]:  # 只看前500条
        msgs = item["messages"]
        dept = "未知"
        for m in msgs:
            # 从 user 内容首行"【科室：xxx】"里抠出科室名
            if m["role"] == "user" and "科室" in m["content"]:
                dept = m["content"].split("：")[1].split("\n")[0] if "：" in m["content"] else "未知"
                break
        if dept not in dept_samples:
            dept_samples[dept] = []
        if len(dept_samples[dept]) < 3:
            dept_samples[dept].append(item)   # 每科室最多 3 条，控制 CPU 推理总时长

    print(f"\n测试科室: {list(dept_samples.keys())}")
    total_samples = sum(len(v) for v in dept_samples.values())
    print(f"测试样本数: {total_samples}")

    # 加载分词器（两个模型共用同一份词表）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 测试原始模型
    print("\n--- 原始模型 (Qwen2.5-0.5B-Instruct) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,   # 与训练时同精度，公平对比
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    base_results = []
    for dept, samples in dept_samples.items():
        for item in samples:
            msgs = item["messages"]
            reference = msgs[-1]["content"]  # assistant的答案 = 参考答案
            prompt = format_prompt(msgs[:-1]) + "<|im_start|>assistant\n"   # 去掉答案段再补生成起点

            response = generate(base_model, tokenizer, prompt)
            bleu = simple_bleu(reference, response)

            base_results.append({
                "department": dept,
                "question": msgs[-2]["content"][:100],
                "reference": reference[:200],
                "response": response[:300],
                "bleu1_char": bleu,
            })
            print(f"  [{dept}] BLEU-1(char)={bleu:.3f} | Q: {msgs[-2]['content'][:60]}...")

    # 释放基础模型
    del base_model   # CPU 全精度 0.5B ≈ 2GB，先释放再载下一组，防内存翻倍
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 测试微调模型
    print("\n--- 微调模型 (LoRA) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)   # 基座挂载 LoRA 旁路
    lora_model.eval()   # 关 dropout，进评估模式

    lora_results = []
    for dept, samples in dept_samples.items():
        for item in samples:
            msgs = item["messages"]
            reference = msgs[-1]["content"]
            prompt = format_prompt(msgs[:-1]) + "<|im_start|>assistant\n"

            response = generate(lora_model, tokenizer, prompt)
            bleu = simple_bleu(reference, response)

            lora_results.append({
                "department": dept,
                "question": msgs[-2]["content"][:100],
                "reference": reference[:200],
                "response": response[:300],
                "bleu1_char": bleu,
            })
            print(f"  [{dept}] BLEU-1(char)={bleu:.3f} | Q: {msgs[-2]['content'][:60]}...")
    
    # 汇总对比
    avg_base_bleu = sum(r["bleu1_char"] for r in base_results) / len(base_results) if base_results else 0
    avg_lora_bleu = sum(r["bleu1_char"] for r in lora_results) / len(lora_results) if lora_results else 0
    
    print("\n" + "=" * 60)
    print("评估总结")
    print(f"  原始模型 平均 BLEU-1(char): {avg_base_bleu:.4f}")
    print(f"  微调模型 平均 BLEU-1(char): {avg_lora_bleu:.4f}")
    print(f"  提升: {(avg_lora_bleu - avg_base_bleu) / max(avg_base_bleu, 0.001) * 100:+.1f}%")
    
    # 保存详细结果
    os.makedirs("eval", exist_ok=True)
    report = {
        "base_model_avg_bleu1": avg_base_bleu,
        "lora_model_avg_bleu1": avg_lora_bleu,
        "improvement_pct": (avg_lora_bleu - avg_base_bleu) / max(avg_base_bleu, 0.001) * 100,
        "base_results": base_results,
        "lora_results": lora_results,
    }
    with open("eval/report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Markdown报告
    with open("eval/report.md", 'w', encoding='utf-8') as f:
        f.write("# 医疗问诊模型评估报告\n\n")
        f.write(f"## 整体指标\n\n")
        f.write(f"| 模型 | 平均 BLEU-1 (字符级) |\n")
        f.write(f"|------|----------------------|\n")
        f.write(f"| 原始 Qwen2.5-0.5B | {avg_base_bleu:.4f} |\n")
        f.write(f"| LoRA微调后 | {avg_lora_bleu:.4f} |\n")
        f.write(f"| **提升** | **{(avg_lora_bleu - avg_base_bleu) / max(avg_base_bleu, 0.001) * 100:+.1f}%** |\n")
        
        f.write(f"\n## 各科室对比\n\n")
        for dept in dept_samples:
            base_dept = [r for r in base_results if r["department"] == dept]
            lora_dept = [r for r in lora_results if r["department"] == dept]
            if base_dept and lora_dept:
                b = sum(r["bleu1_char"] for r in base_dept) / len(base_dept)
                l = sum(r["bleu1_char"] for r in lora_dept) / len(lora_dept)
                f.write(f"- **{dept}**: {b:.4f} → {l:.4f}\n")
        
        f.write(f"\n## 样例对比\n\n")
        for i, (base_r, lora_r) in enumerate(zip(base_results[:5], lora_results[:5])):
            f.write(f"### 样例 {i+1}: {base_r['department']}\n")
            f.write(f"**问题**: {base_r['question'][:100]}...\n\n")
            f.write(f"**参考答案**: {base_r['reference'][:150]}...\n\n")
            f.write(f"**原始模型**: {base_r['response'][:150]}...\n\n")
            f.write(f"**微调模型**: {lora_r['response'][:150]}...\n\n")
            f.write("---\n\n")
    
    print(f"\n评估报告已保存到 eval/report.md")


if __name__ == "__main__":
    main()
