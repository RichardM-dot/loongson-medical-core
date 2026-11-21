#!/usr/bin/env python3
"""Q4_K量化 FP16 GGUF → Q4_K GGUF (Qwen2.5-0.5B hidden_size=896, K-quant不兼容)"""

import os, json, struct
import numpy as np
from pathlib import Path
from gguf.gguf_reader import GGUFReader
from gguf import GGUFWriter, LlamaFileType, GGMLQuantizationType
from gguf.quants import quantize as gguf_quantize

GGUF_FP16 = "qwen2.5-0.5b-f16.gguf"   # 上游输入：quantize_model.py 的中间产物
GGUF_Q = "/home/richard/qwen2.5-0.5b-q4_k_m.gguf"   # 最终部署产物

# Qwen2.5-0.5B hidden_size=896, 896%256=128 所以 K-quant 无法使用
# Q4_K block=256, 896%256≠0, Python gguf库不兼容
# Q4_K: ~4.5 bpw, 体积约为 FP16 的 1/4

# 本脚本是 quantize_model.py 中 quantize_q4k() 的独立重实现：
# 手工逐字段搬元数据（add_key_value），不依赖 add_field 的批量接口，
# 用于绕开旧版 gguf 库对 Qwen2.5-0.5B 尺寸的 K-quant 兼容问题
QTYPE = GGMLQuantizationType.Q4_K
FTYPE = LlamaFileType.MOSTLY_Q4_K


# ── 把 FP16 文件的 KV 元数据逐字段搬到新 writer（file_type 等结构性字段除外）──
def copy_metadata(reader, writer):
    # skip 列表 = 由 writer 自己重新生成的字段：
    # 头部计数字段 + 架构名 + 量化类型标记，搬旧值反而会自相矛盾
    skip = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count',
            'general.file_type', 'general.quantization_version',
            'general.alignment', 'general.architecture'}

    for name, field in reader.fields.items():
        if name in skip:
            continue

        # field.data 是"数据片段索引"列表，真正的值在 field.parts[data_idx]
        for i, data_idx in enumerate(field.data):
            vtype = field.types[i]
            val = field.parts[data_idx]
            if isinstance(val, np.ndarray):
                if val.dtype.kind == 'S' or val.dtype.kind == 'U':
                    # 字节串 → UTF-8 字符串，并去掉 GGUF 字符串补位的 '\0'
                    val = val.tobytes().decode('utf-8', errors='replace').rstrip('\x00')
                else:
                    val = val.item() if val.size == 1 else val   # 单元素数组还原成标量
            elif isinstance(val, np.generic):
                val = val.item()   # numpy 标量 → Python 原生类型

            sub_type = None
            if hasattr(field, 'sub_types') and i < len(field.sub_types):
                sub_type = field.sub_types[i]   # 数组元素的子类型，保证写回类型一致

            writer.add_key_value(name, val, vtype, sub_type)

    writer.add_file_type(FTYPE)
    writer.add_quantization_version(2)


# ── 主流程：读 FP16 GGUF → 搬元数据 → 逐张量 Q4_K 量化 → 三段式写出 ──
def main():
    print(f"Loading FP16 GGUF...")
    reader = GGUFReader(GGUF_FP16)

    # 从元数据里解出架构名（GGUF 字符串是 bytes 数组，需手工解码）
    arch_field = reader.get_field('general.architecture')
    arch_name = "qwen2"
    if arch_field:
        for p in arch_field.parts:
            if isinstance(p, np.ndarray) and (p.dtype.kind == 'S' or p.dtype.kind == 'U'):
                arch_name = p.tobytes().decode('utf-8', errors='replace').rstrip('\x00')
                break

    print(f"Architecture: {arch_name}, Tensors: {len(reader.tensors)}")
    print(f"Quantization: {QTYPE.name} (~4.5 bpw)")

    output_path = Path(GGUF_Q)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = GGUFWriter(str(output_path), arch_name)

    print("Copying metadata...")
    copy_metadata(reader, writer)

    print("Quantizing tensors...")
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        data = tensor.data.astype(np.float32)   # 量化器统一吃 fp32
        shape = tensor.shape

        # 与 quantize_model.py 同一判定：≥2 维且最大边 ≥32 才量化，
        # 一维 norm/bias 与小张量保持 F16 精度无损
        is_weight = len(shape) >= 2 and max(shape) >= 32

        if is_weight:
            try:
                quantized = gguf_quantize(data, QTYPE)
                writer.add_tensor(name, quantized, raw_dtype=QTYPE)
            except Exception as e:
                print(f"  Warning: {name} ({shape}) quantize failed ({e}), using F16")   # 逐张量降级，不中断
                writer.add_tensor(name, data.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, data.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(reader.tensors)}")

    print("Writing GGUF file...")
    writer.write_header_to_file()   # 头
    writer.write_kv_data_to_file()  # 元数据
    writer.write_tensors_to_file()  # 张量数据
    writer.close()

    orig_mb = os.path.getsize(GGUF_FP16) / 1024 / 1024
    final_mb = os.path.getsize(str(output_path)) / 1024 / 1024
    print(f"\nDone! {orig_mb:.1f} MB → {final_mb:.1f} MB ({final_mb/orig_mb*100:.1f}%)")
    print(f"Output: {GGUF_Q}")


if __name__ == "__main__":
    main()
