#!/usr/bin/env python3
"""数据预处理: 合并医疗对话CSV → Qwen Chat模板JSON → train/val/test"""

import os
import json
import csv
import random
from pathlib import Path

# 固定随机种子：保证 shuffle 与数据划分可复现（同样的输入永远得到同样的 train/val/test）
random.seed(42)

# 原始 CSV 数据集目录（Chinese-medical-dialogue，需手动放到脚本同级）
DATA_DIR = Path("shujuji/Chinese-medical-dialogue-data-master/Data_数据")
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)   # 输出 train/val/test.json 的目录

# 科室配置: (文件夹名, csv文件名, 科室中文名)
# 每个 CSV 是该科室独立文件，文件夹名即 GitHub 仓库原始命名
DEPARTMENTS = [
    ("Andriatria_男科", "男科5-13000.csv", "男科"),
    ("IM_内科", "内科5000-33000.csv", "内科"),
    ("OAGD_妇产科", "妇产科6-28000.csv", "妇产科"),
    ("Oncology_肿瘤科", "肿瘤科5-10000.csv", "肿瘤科"),
    ("Pediatric_儿科", "儿科5-14000.csv", "儿科"),
    ("Surgical_外科", "外科5-14000.csv", "外科"),
]

# 注入每条样本 system 角色的统一人设：约束模型输出医疗建议 + 不确定时引导就医
SYSTEM_PROMPT = "你是一位专业的医疗问诊助手，请根据患者的病情描述给出专业、准确的医学建议。如果你不确定，请建议患者去医院就诊。"


def clean_text(text: str) -> str:
    """清理文本"""
    # 换行/回车压成空格：对话 QA 保持单行，方便后续按条切分与统计
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    # 去除多余空格
    # join(split()) 一举三得：去首尾空白、压连续空格为 1 个、去全角空格外的制表符
    text = ' '.join(text.split())
    return text


def format_qwen_chat(question: str, answer: str, department: str) -> dict:
    """格式化为Qwen Chat模板"""
    # 科室标签放进 user 内容首行：推理时可按科室检索，也让模型显式感知科室语境
    user_content = f"【科室：{department}】\n{question}"

    # 输出结构 = OpenAI messages 风格，与 Qwen 的 ChatML 三角色一一对应
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
    }


def read_csv_safe(filepath: Path) -> list:
    """安全读取CSV，处理各种格式问题"""
    samples = []

    # 尝试多种编码：上游数据来自多人整理，编码不统一是常态；
    # latin-1 放最后兜底（任何字节流都能解，保证函数有返回内容）
    content = None
    for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue

    if content is None:
        print(f"  Error: {filepath.name} 无法解码")
        return samples

    # 先整体读入内存再 StringIO：避免编码重试时反复开文件
    import io
    f = io.StringIO(content)
    reader = csv.reader(f)
    header = next(reader, None)

    if header is None:
        return samples

    # 找到列索引
    try:
        # 列名可能是 department,title,ask,answer 或 department,title,question,answer
        header_lower = [h.lower().strip() for h in header]

        dept_idx = None
        ask_idx = None
        ans_idx = None

        # 按同义词表匹配列名，容忍上游列名差异（ask/question/input 等）
        for i, h in enumerate(header_lower):
            if h in ('department', 'depart'):
                dept_idx = i
            elif h in ('ask', 'question', 'input'):
                ask_idx = i
            elif h in ('answer', 'output', 'response'):
                ans_idx = i

        if ask_idx is None or ans_idx is None:
            print(f"  Warning: {filepath.name} 无法识别列名: {header}")
            return samples

        for row in reader:
            if len(row) <= max(ask_idx, ans_idx):
                continue   # 列数不足的脏行直接跳过

            question = clean_text(row[ask_idx])
            answer = clean_text(row[ans_idx])
            department = clean_text(row[dept_idx]) if dept_idx is not None and dept_idx < len(row) else ""

            # 过滤太短或太长的样本：
            # 短样本无医学信息量，长样本会撑爆 512 token 训练窗口（见 train_lora.py）
            if len(question) < 5 or len(answer) < 10:
                continue
            if len(question) > 500 or len(answer) > 1000:
                continue

            samples.append({
                "question": question,
                "answer": answer,
                "department": department
            })
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")

    return samples


# ── 流水线入口：读 CSV -> 统一科室名 -> 按科室分层划分 -> ChatML 格式落盘 ──
def main():
    all_samples = []

    print("读取各科室数据...")
    for folder, csv_file, dept_name in DEPARTMENTS:
        csv_path = DATA_DIR / folder / csv_file
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} 不存在")   # 数据集没放全时降级为部分科室训练
            continue

        samples = read_csv_safe(csv_path)
        # 统一科室名：CSV 内 department 列常有空值/别名，兜底用配置里的中文名
        for s in samples:
            if not s["department"]:
                s["department"] = dept_name

        print(f"  {dept_name}: {len(samples)} 条有效样本")
        all_samples.extend(samples)

    print(f"\n总计: {len(all_samples)} 条有效样本")

    # 按科室分层采样：先按科室分桶再在桶内切分，
    # 保证 train/val/test 三份里每个科室比例一致（防止某科室全漏进测试集）
    from collections import defaultdict
    dept_samples = defaultdict(list)
    for s in all_samples:
        dept_samples[s["department"]].append(s)

    train_samples = []
    val_samples = []
    test_samples = []

    for dept, samples in dept_samples.items():
        random.shuffle(samples)   # seed=42 已固定，划分结果可复现
        n = len(samples)
        n_train = int(n * 0.9)    # 90% 训练
        n_val = int(n * 0.05)     # 5% 验证（选最优 checkpoint 用）

        # 剩余 ~5% 自然成为测试集（evaluate.py 消费）
        train_samples.extend(samples[:n_train])
        val_samples.extend(samples[n_train:n_train + n_val])
        test_samples.extend(samples[n_train + n_val:])

    print(f"划分: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # 格式化为Qwen Chat并保存
    for name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        formatted = []
        for s in samples:
            formatted.append(format_qwen_chat(s["question"], s["answer"], s["department"]))

        out_path = OUT_DIR / f"{name}.json"
        # ensure_ascii=False 保留中文原文字符；indent=2 便于人工抽查
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        print(f"  {name}.json: {len(formatted)} 条 → {out_path}")


if __name__ == "__main__":
    main()
