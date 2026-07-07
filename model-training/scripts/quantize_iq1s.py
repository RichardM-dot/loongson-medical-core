#!/usr/bin/env python3
"""Q4_K量化 FP16 GGUF → Q4_K GGUF (Qwen2.5-0.5B hidden_size=896, K-quant不兼容)"""

import os, json, struct
import numpy as np
from pathlib import Path
from gguf.gguf_reader import GGUFReader
from gguf import GGUFWriter, LlamaFileType, GGMLQuantizationType
from gguf.quants import quantize as gguf_quantize

GGUF_FP16 = "qwen2.5-0.5b-f16.gguf"
GGUF_Q = "/home/richard/qwen2.5-0.5b-q4_k_m.gguf"

# Qwen2.5-0.5B hidden_size=896, 896%256=128 所以 K-quant 无法使用
# Q4_K block=256, 896%256≠0, Python gguf库不兼容
# Q4_K: ~4.5 bpw, 体积约为 FP16 的 1/4

QTYPE = GGMLQuantizationType.Q4_K
FTYPE = LlamaFileType.MOSTLY_Q4_K


def copy_metadata(reader, writer):
    skip = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count',
            'general.file_type', 'general.quantization_version',
            'general.alignment', 'general.architecture'}
    
    for name, field in reader.fields.items():
        if name in skip:
            continue
        
        for i, data_idx in enumerate(field.data):
            vtype = field.types[i]
            val = field.parts[data_idx]
            if isinstance(val, np.ndarray):
                if val.dtype.kind == 'S' or val.dtype.kind == 'U':
                    val = val.tobytes().decode('utf-8', errors='replace').rstrip('\x00')
                else:
                    val = val.item() if val.size == 1 else val
            elif isinstance(val, np.generic):
                val = val.item()
            
            sub_type = None
            if hasattr(field, 'sub_types') and i < len(field.sub_types):
                sub_type = field.sub_types[i]
            
            writer.add_key_value(name, val, vtype, sub_type)
    
    writer.add_file_type(FTYPE)
    writer.add_quantization_version(2)


def main():
    print(f"Loading FP16 GGUF...")
    reader = GGUFReader(GGUF_FP16)
    
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
        data = tensor.data.astype(np.float32)
        shape = tensor.shape
        
        is_weight = len(shape) >= 2 and max(shape) >= 32
        
        if is_weight:
            try:
                quantized = gguf_quantize(data, QTYPE)
                writer.add_tensor(name, quantized, raw_dtype=QTYPE)
            except Exception as e:
                print(f"  Warning: {name} ({shape}) quantize failed ({e}), using F16")
                writer.add_tensor(name, data.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        else:
            writer.add_tensor(name, data.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(reader.tensors)}")
    
    print("Writing GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    orig_mb = os.path.getsize(GGUF_FP16) / 1024 / 1024
    final_mb = os.path.getsize(str(output_path)) / 1024 / 1024
    print(f"\nDone! {orig_mb:.1f} MB → {final_mb:.1f} MB ({final_mb/orig_mb*100:.1f}%)")
    print(f"Output: {GGUF_Q}")


if __name__ == "__main__":
    main()
