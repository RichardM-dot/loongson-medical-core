#!/usr/bin/env python3
"""评估微调效果: 对比微调前后模型在医疗问诊上的表现"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

MODEL_PATH = "."
LORA_PATH = "lora_adapter"
TEST_DATA = "data/test.json"
NUM_SAMPLES = 10  # 测试样本数


def format_prompt(messages: list) -> str:
    """构建Qwen Chat格式prompt"""
    text_parts = []
    for msg in messages:
        text_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    prompt = "\n".join(text_parts)
    # 只保留到assistant开始
    return prompt


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 只提取新生成的部分
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def simple_bleu(reference: str, candidate: str) -> float:
    """简单BLEU-1计算 (字符级别)"""
    ref_chars = set(reference)
    cand_chars = set(candidate)
    if not cand_chars:
        return 0.0
    overlap = len(ref_chars & cand_chars)
    return overlap / len(cand_chars)


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
            if m["role"] == "user" and "科室" in m["content"]:
                dept = m["content"].split("：")[1].split("\n")[0] if "：" in m["content"] else "未知"
                break
        if dept not in dept_samples:
            dept_samples[dept] = []
        if len(dept_samples[dept]) < 3:
            dept_samples[dept].append(item)
    
    print(f"\n测试科室: {list(dept_samples.keys())}")
    total_samples = sum(len(v) for v in dept_samples.values())
    print(f"测试样本数: {total_samples}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 测试原始模型
    print("\n--- 原始模型 (Qwen2.5-0.5B-Instruct) ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    base_results = []
    for dept, samples in dept_samples.items():
        for item in samples:
            msgs = item["messages"]
            reference = msgs[-1]["content"]  # assistant的答案
            prompt = format_prompt(msgs[:-1]) + "<|im_start|>assistant\n"
            
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
    del base_model
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
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_model.eval()
    
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
