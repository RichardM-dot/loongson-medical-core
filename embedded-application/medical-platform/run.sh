#!/bin/bash
# 龙芯智慧医疗平台 启动脚本
cd /home/richard/医疗平台
export LD_LIBRARY_PATH=/home/richard/sense_voice_demo/lib:$LD_LIBRARY_PATH
python3 medical_platform.py
