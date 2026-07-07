# Loongson Medical Core

龙芯医疗 AI 核心平台 —— 基于 Loongson（龙芯）架构的嵌入式医疗辅助系统。

## 项目结构

```
loongson-medical-core/
├── embedded-application/     # 嵌入式应用（龙芯医疗助手）
│   ├── assistant.c           # C 语言医疗助手核心
│   └── medical-platform/     # Python 医疗平台 GUI
├── inference-engine/         # 推理引擎
│   ├── qwen2.5-0.5b-instruct-q4_k_m.gguf   # 量化医疗对话模型
│   └── sense-voice-small-q4_k.gguf         # 语音识别模型
├── model-training/           # 模型训练
│   ├── data/                 # 训练/验证/测试数据（中文医疗对话）
│   ├── models/               # 基座模型 + LoRA + 合并模型
│   ├── scripts/              # 训练/评估/量化脚本
│   └── outputs/              # 评估报告
└── speech-engine/            # 语音引擎
    ├── sense-voice-main      # SenseVoice 语音识别
    └── sense-voice-small-q4_k.gguf  # 量化语音模型
```

## 技术栈

- **推理**: llama.cpp / Qwen2.5-0.5B-Instruct（Q4_K_M 量化）
- **语音**: SenseVoice-Small（Q4_K 量化）
- **训练**: LoRA 微调 + IQ1S 量化
- **平台**: 龙芯架构嵌入式 Linux / Python 3 + GTK
- **数据**: 中文医疗对话数据集（内科/外科/儿科/妇产科/男科/肿瘤科）

## 快速开始

```bash
# 推理引擎
cd inference-engine
./llama-cli -m qwen2.5-0.5b-instruct-q4_k_m.gguf -p "你好"

# 医疗平台
cd embedded-application/medical-platform
bash run.sh
```

## License

See [LICENSE](model-training/LICENSE)
