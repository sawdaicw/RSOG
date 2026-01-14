面向开放指令的多源遥感多模态模型

一、项目简介
# 本项目是一个基于Qwen2.5-VL模型的多模态视觉语言理解与生成系统，专注于遥感图像的分析和处理。项目通过微调训练和评估流程，实现对遥感图像中目标的检测、识别和描述生成。
  该项目旨在解决传统遥感检测模型仅能识别固定类别的局限性，重点探索开放式指令下的目标检测任务。通过引入动态指令处理机制，模型能够理解复杂的自然语言任务描述，并将其转化为精确的检测行为，从而在复杂的无人机/卫星图像中实现高精度的通用目标检测。

# 核心亮点与机制
  指令动态匹配机制: 针对用户输入的模糊或复杂指令，项目设计了一个专门的指令处理器。它利用 LLM 的能力对原始指令进行：
  意图分类 : 判断指令是明确（Explicit）、隐含（Implicit）还是模糊（Ambiguous），并评估其难度与置信度。
  任务分解 : 将复杂的场景描述分解为具体的“检测目标（Targets）”和“环境约束（Environment）”。
  智能路由 : 仅在必要时（如指令复杂且置信度高时）重写指令，简单明确的指令则保持原样，以平衡效率与准确率。

# 应用场景
  遥感图像智能解译
  地理信息系统分析
  环境监测与评估
  城市规划与管理

二、环境部署
# 环境依赖
  设置环境
  conda create -n Qwen2.5 python=3.11 -y
  conda activate Qwen2.5

  安装依赖
  pip install vllm
  pip install git+https://github.com/huggingface/transformers
  pip install torch accelerate
  
# 文件迁移
  以本文件夹作为母文件夹
  将模型文件export_v5_11968放到./Qwen2.5-VL-FT-Remote
  结果results放到./Qwen2.5-VL-FT-Remote/results
  数据集datasets（包含VLAD_R和VRSBench两部分）放到./Qwen2.5-VL-FT-Remote/datasets

# 路径修改
  分别运行
  python ./tools/dataset_Remote/02_spilt_abspath.py
  python ./tools/dataset_VRSBench/02_abspath.py
  python ./tools/generate_eval_instructions.py

# 模型部署
  CUDA_VISIBLE_DEVICES=0 vllm serve （注意修改模型路径）./export_v5_11968 \
      --port 8015 \
    （--gpu-memory-utilization 0.9 \
      --max-model-len 8192 \
      --dtype bfloat16）

三、实验
  #检测实验
  python ./tools/dataset_Remote/05_eval.py 记得调bbox detector的模型api

  # 处理fixed任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name ./export_v5_11968 --task_type fixed

  # 处理open任务
    python ./tools/dataset_Remote/06_eval_statistics.py --file_name ./export_v5_11968 --task_type open
    (注意修改模型路径)

  # 处理open ended任务,注意修改文件路径
    python ./tools/dataset_Remote/07_eval_statistics_open_ended.py

  # 启用指令动态匹配机制处理open-ended任务
    1.检测
      首先修改instruction.py与eval_instruction.py中的模型api
      再修改eval_instruction.py中的图片路径（使用绝对路径）
      python ./tools/dataset_Remote/010_eval_instruction.py

    2.将结果转化为coco格式
      注意修改需要处理的json文件路径
      python ./tools/dataset_Remote/07_eval_statistics_open_ended.py 

    3.可视化
      注意修改图片路径，确保该图片启用与未启用的coco结果均已保存在对应路径下
      python ./tools/dataset_Remote/011_case_study.py