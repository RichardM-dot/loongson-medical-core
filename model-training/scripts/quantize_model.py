#!/usr/bin/env python3
"""合并LoRA → GGUF转换 → Q4_K量化"""

import os, json
import torch, numpy as np
from pathlib import Path
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gguf
from gguf import GGUFWriter, LlamaFileType, GGMLQuantizationType
from gguf.constants import Keys, MODEL_ARCH
from gguf.tensor_mapping import TensorNameMap
from gguf.quants import quantize as gguf_quantize

BASE_MODEL = "."
LORA_PATH = "lora_adapter"
MERGED_DIR = "merged_model"
GGUF_FP16 = "qwen2.5-0.5b-f16.gguf"
GGUF_Q4_K = "/home/richard/qwen2.5-0.5b-q4_k_m.gguf"

K = Keys  # shortcut


def merge_lora():
    print("[1/4] 合并 LoRA...")
    if os.path.exists(os.path.join(MERGED_DIR, "model.safetensors")):
        print("  已存在, 跳过")
        return
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_DIR, safe_serialization=True)
    import shutil
    for f in ["config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = os.path.join(BASE_MODEL, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(MERGED_DIR, f))
    print("  合并完成")


def convert_to_gguf():
    print("[2/4] 转换 HF → GGUF FP16...")
    if os.path.exists(GGUF_FP16):
        print(f"  已存在, 跳过")
        return
    
    with open(os.path.join(MERGED_DIR, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(MERGED_DIR, "tokenizer_config.json")) as f:
        tok = json.load(f)
    
    n_layers = cfg["num_hidden_layers"]
    writer = GGUFWriter(GGUF_FP16, "qwen2")
    
    # 元数据
    writer.add_uint32(K.Attention.HEAD_COUNT, cfg["num_attention_heads"])
    writer.add_uint32(K.Attention.HEAD_COUNT_KV, cfg["num_key_value_heads"])
    writer.add_uint32(K.LLM.BLOCK_COUNT, n_layers)
    writer.add_uint32(K.LLM.EMBEDDING_LENGTH, cfg["hidden_size"])
    writer.add_uint32(K.LLM.FEED_FORWARD_LENGTH, cfg["intermediate_size"])
    writer.add_float32(K.Attention.LAYERNORM_RMS_EPS, cfg["rms_norm_eps"])
    writer.add_uint32(K.Rope.DIMENSION_COUNT, 128)
    writer.add_float32(K.Rope.FREQ_BASE, cfg["rope_theta"])
    writer.add_uint32(K.LLM.CONTEXT_LENGTH, cfg["max_position_embeddings"])
    writer.add_uint32(K.LLM.VOCAB_SIZE, cfg["vocab_size"])
    writer.add_bool(K.LLM.USE_PARALLEL_RESIDUAL, False)
    
    writer.add_uint32(K.Tokenizer.BOS_ID, cfg.get("bos_token_id", 151643))
    writer.add_uint32(K.Tokenizer.EOS_ID, cfg.get("eos_token_id", 151645))
    writer.add_uint32(K.Tokenizer.PAD_ID, tok.get("pad_token_id", 151643))
    
    writer.add_uint32(K.General.FILE_TYPE, LlamaFileType.MOSTLY_F16)
    writer.add_uint32(K.General.QUANTIZATION_VERSION, gguf.GGML_QUANT_VERSION)
    
    # 加载权重
    model_file = os.path.join(MERGED_DIR, "model.safetensors")
    tmap = TensorNameMap(MODEL_ARCH.QWEN2, n_layers)
    
    with safe_open(model_file, framework="pt", device="cpu") as f:
        tensor_names = list(f.keys())
    
    print(f"  转换 {len(tensor_names)} 个张量...")
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for i, hf_name in enumerate(tensor_names):
            gguf_name = tmap.get_name(hf_name, try_suffixes=(".weight", ".bias"))
            if gguf_name is None:
                continue
            tensor = f.get_tensor(hf_name)
            tensor_f16 = tensor.to(torch.float16).numpy() if tensor.dtype != torch.float16 else tensor.numpy()
            writer.add_tensor(gguf_name, tensor_f16)
            if (i + 1) % 50 == 0:
                print(f"    进度: {i+1}/{len(tensor_names)}")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_mb = os.path.getsize(GGUF_FP16) / 1024 / 1024
    print(f"  GGUF FP16 完成: {size_mb:.1f} MB")


def quantize_q4k():
    print("[3/4] Q4_K 量化...")
    
    output_path = Path(GGUF_Q4_K)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    from gguf.gguf_reader import GGUFReader
    reader = GGUFReader(GGUF_FP16)
    arch_field = reader.get_field('general.architecture')
    arch = str(arch_field.parts[-1][-1]) if arch_field else "qwen2"
    writer = GGUFWriter(str(output_path), arch)
    
    # 复制元数据
    for field in reader.fields.values():
        if field.name != K.General.FILE_TYPE:
            writer.add_field(field)
    writer.add_uint32(K.General.FILE_TYPE, LlamaFileType.MOSTLY_Q4_K)
    writer.add_uint32(K.General.QUANTIZATION_VERSION, gguf.GGML_QUANT_VERSION)
    
    tensors = list(reader.tensors.items())
    print(f"  量化 {len(tensors)} 个张量...")
    
    for i, (name, tensor) in enumerate(tensors):
        data = tensor.data
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        
        is_weight = data.ndim >= 2 and max(data.shape) >= 32
        
        if is_weight:
            try:
                qtype = GGMLQuantizationType.Q4_K
                quantized = gguf_quantize(data, qtype)
                writer.add_tensor(name, quantized, raw_dtype=qtype)
            except Exception as e:
                print(f"    Warning: {name} 量化失败, 降级F16 ({e})")
                writer.add_tensor(name, data.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, data.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        
        if (i + 1) % 50 == 0:
            print(f"    进度: {i+1}/{len(tensors)}")
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size_mb = os.path.getsize(str(output_path)) / 1024 / 1024
    orig_mb = os.path.getsize(GGUF_FP16) / 1024 / 1024
    print(f"  Q4_K 完成: {orig_mb:.1f}MB → {size_mb:.1f}MB ({size_mb/orig_mb*100:.1f}%)")


def verify():
    print("[4/4] 验证...")
    output_path = Path(GGUF_Q4_K)
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  ✓ {GGUF_Q4_K} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {GGUF_Q4_K} 不存在!")
        return
    
    from gguf.gguf_reader import GGUFReader
    reader = GGUFReader(str(output_path))
    arch_field = reader.get_field('general.architecture')
    arch = str(arch_field.parts[-1][-1]) if arch_field else "unknown"
    n_tensors = len(reader.tensors)
    print(f"  ✓ 架构: {arch}, 张量数: {n_tensors}")
    
    ft_field = reader.get_field('general.file_type')
    if ft_field:
        print(f"  ✓ 文件类型: {ft_field.parts[-1]}")
    
    print("  验证通过!")


if __name__ == "__main__":
    merge_lora()
    convert_to_gguf()
    quantize_q4k()
    verify()
    print(f"\n最终模型: {GGUF_Q4_K}")
    print("完成!")
