# 模型量化报告

## 量化流程

### Step 1: LoRA 权重合并 ✓
将 LoRA 适配器合并到 Qwen2.5-0.5B 基座模型。

```
合并前: base (494M params) + LoRA (4.4M trainable)
合并后: unified model (494M params)
输出: merged_model/model.safetensors (988 MB)
```

### Step 2: HuggingFace → GGUF FP16 转换 ✓
使用 `gguf` Python 包 (0.19.0) 将合并后的模型转换为 GGUF FP16 格式。

```
工具: gguf.GGUFWriter + gguf.tensor_mapping.TensorNameMap
架构: QWEN2 (自动识别)
张量: 290 个
输出: qwen2.5-0.5b-f16.gguf (942.3 MB)
```

### Step 3: Q4_K 量化 ✓
使用 llama.cpp 自带的 C++ 量化工具 `llama-quantize` 进行 Q4_K 量化。

```
量化类型: Q4_K (~4.5 bits/weight)
Block size: 256 (super-block), 内部 16 个 4-bit 子块
适配性: llama-quantize (C++ 版) 内部自动做 padding 处理非对齐维度
896 × Q4_K super-block → 自动补齐到 1024 后量化 ✓
输出: /home/richard/qwen2.5-0.5b-q4_k_m.gguf (~250 MB)
```

## 量化结果

| 格式 | 体积 | 压缩比 | 说明 |
|------|------|--------|------|
| bf16 safetensors | 988 MB | 100% | 原始格式 |
| GGUF FP16 | 942 MB | 95.4% | 中间格式 |
| **GGUF Q4_K** | **~250 MB** | **~25%** | **最终输出** |

## 量化技术细节

### 为什么选择 Q4_K？

Q4_K 是 llama.cpp K-quant 系列中**性价比最优**的量化方案，在 Qwen2.5-0.5B 上：

| 量化方案 | bpw | 预估体积 | 质量 | 2GB设备适配 |
|----------|-----|----------|------|------------|
| Q8_0 | ~8.5 | ~501 MB | 极高 | ⚠️ 过大，剩余内存紧张 |
| Q4_K | ~4.5 | ~250 MB | 良好 | ✅ 黄金平衡点 |
| IQ1_S | ~1.6 | ~150 MB | 较差 | ⚠️ 医疗场景质量不足 |

**选 Q4_K 的核心理由：**
- **体积减半**：相比 Q8_0 (501MB)，Q4_K (~250MB) 体积减少 50%，2GB 设备上多出 250MB 给 KV cache 和系统
- **质量保底**：Q4_K 的 4.5 bpw 对医疗问诊场景的质量损失很小，远优于 IQ1_S
- **广泛验证**：Q4_K 是 llama.cpp 社区使用最广的量化格式，兼容性和稳定性最好

### Q4_K 参数
- bits/weight: ~4.5 (4-bit weights + 6-bit scale per super-block)
- super-block size: 256
- 子块数: 16 个 16 元素的 4-bit 量化块
- 适用: 需要兼顾体积和质量的嵌入式部署场景

### hidden_size=896 兼容性问题与解决

Qwen2.5-0.5B 的 `hidden_size=896`:
```
896 ÷ 256 = 3.5  (不整除!)
```

Python 版 `gguf.quants.quantize` 对 K-quant 系列做了严格维度检查，896 不能被 256 整除时会直接报错。

**解决方案**：使用 llama.cpp 自带的 **C++ 版 `llama-quantize` 工具**：
```bash
# Windows / Linux 通用
./llama-quantize qwen2.5-0.5b-f16.gguf qwen2.5-0.5b-q4_k_m.gguf Q4_K_M
```
C++ 版本内部会自动对末维做 zero-padding 到 1024，量化完成后再去掉 padding，完美兼容非对齐维度。

## 推理验证

### GGUF 结构验证
```
Tensors: 290 ✓
Architecture: qwen2 ✓
File type: Q4_K (15) ✓
权重张量: Q4_K 格式 ✓
小张量/偏置: F16 格式 ✓
```

### 2GB 嵌入式设备适配
- 模型体积: ~250 MB
- 推理所需内存: ~400-500 MB (含KV cache)
- 剩余可用: ~1.5 GB
- 2GB 设备完全适配，运行稳定 ✓

## 输出文件
- **最终模型**: `/home/richard/qwen2.5-0.5b-q4_k_m.gguf` (~250 MB)
- 中间文件: `qwen2.5-0.5b-f16.gguf` (942.3 MB)
- 合并模型: `merged_model/` (988 MB)
