# 面向开放指令的多源遥感多模态模型

## 一、项目简介

本项目是一个**基于 Qwen2.5-VL 的多模态视觉-语言理解与生成系统**，面向多源遥感图像（卫星 / 无人机）场景，重点研究**开放式自然语言指令下的目标检测与理解任务**。

不同于传统遥感检测模型只能识别**固定类别、固定任务形式**的问题，本项目通过对 Qwen2.5-VL 进行微调与系统化工程改造，使模型能够：

* 理解**复杂、模糊或隐含的自然语言指令**
* 将开放式语言任务自动转化为**可执行的检测目标与约束条件**
* 在复杂遥感场景中完成**通用目标检测、识别与语义描述生成**

项目的核心目标是：

> **构建一个面向开放指令、可泛化、多任务统一的遥感多模态模型框架。**

---

## 二、核心机制与技术亮点

### 1. 指令动态匹配机制（Instruction Dynamic Matching）

针对遥感场景中用户指令常见的**模糊性、多目标性与高层语义描述问题**，本项目设计了一套基于 LLM 的指令动态处理机制，用于在推理阶段对输入指令进行自适应优化。

该机制包含以下关键步骤：

#### （1）意图分类（Intent Classification）

利用大语言模型对原始指令进行语义分析，判断其类型与可执行性：

* **Explicit（明确指令）**：检测目标清晰、边界明确
* **Implicit（隐含指令）**：目标需从场景语义中推断
* **Ambiguous（模糊指令）**：存在歧义或信息不足

同时评估指令的**复杂度与语义置信度**，为后续处理提供依据。

#### （2）任务分解（Task Decomposition）

将复杂指令拆解为结构化表示：

* **Targets（检测目标）**：需要检测或关注的对象
* **Environment（环境约束）**：空间、语义或上下文限制条件

该过程有效降低了端到端直接理解复杂指令所带来的不稳定性。

#### （3）智能路由与重写（Smart Routing）

* 对于**简单且明确的指令**：直接送入模型，避免不必要的计算开销
* 对于**复杂但高置信度的指令**：触发指令重写与规范化流程

在**推理效率与检测精度之间取得平衡**，避免“一刀切”的指令重写策略。

---

## 三、应用场景

本项目可广泛应用于以下遥感与地理信息场景：

* 遥感图像智能解译
* 地理信息系统（GIS）分析
* 环境监测与变化评估
* 城市规划与精细化管理
* 无人机自主感知与任务理解

---

## 四、环境部署

### 1. 环境依赖

* Python >= 3.11
* CUDA + GPU（推荐 24GB 及以上显存）

### 2. 创建与激活 Conda 环境

```bash
conda create -n Qwen2.5 python=3.11 -y
conda activate Qwen2.5
```

### 3. 安装依赖

```bash
pip install vllm
pip install git+https://github.com/huggingface/transformers
pip install torch accelerate
```

---

## 五、数据与文件结构配置

### 1. 文件迁移说明

以当前仓库根目录作为母文件夹，按如下结构放置文件：

```
Qwen2.5-VL-FT-Remote/
├── export_v5_11968/          # 微调后的模型权重
├── results/                  # 推理与评测结果
├── datasets/                 # 数据集
│   ├── VLAD_R/
│   └── VRSBench/
```

### 2. 路径修正（必须执行）

```bash
python ./tools/dataset_Remote/02_spilt_abspath.py
python ./tools/dataset_VRSBench/02_abspath.py
python ./tools/generate_eval_instructions.py
```

---

## 六、模型部署（vLLM）

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./export_v5_11968 \
  --port 8015 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --dtype bfloat16
```

> ⚠️ 注意：请根据实际情况修改模型路径与 GPU 配置

---

## 七、实验与评测流程

### 1. 检测实验（基础推理）

```bash
python ./tools/dataset_Remote/05_eval.py
```

> 需提前修改 bbox detector 所使用的模型 API

---

### 2. Fixed 任务评测

```bash
python ./tools/dataset_Remote/06_eval_statistics.py \
  --file_name ./export_v5_11968 \
  --task_type fixed
```

---

### 3. Open 任务评测

```bash
python ./tools/dataset_Remote/06_eval_statistics.py \
  --file_name ./export_v5_11968 \
  --task_type open
```

---

### 4. Open-ended 任务评测（无指令动态匹配）

```bash
python ./tools/dataset_Remote/07_eval_statistics_open_ended.py
```

> 注意修改对应的文件路径

---

## 八、启用指令动态匹配机制（Open-ended）

### Step 1：推理检测

1. 修改 `instruction.py` 与 `eval_instruction.py` 中的模型 API
2. 修改 `eval_instruction.py` 中的图片路径（需使用绝对路径）

```bash
python ./tools/dataset_Remote/010_eval_instruction.py
```

---

### Step 2：结果转 COCO 格式

```bash
python ./tools/dataset_Remote/07_eval_statistics_open_ended.py
```

> 注意修改待处理 JSON 文件路径

---

### Step 3：结果可视化与对比分析

```bash
python ./tools/dataset_Remote/011_case_study.py
```

> 确保启用 / 未启用指令动态匹配机制的 COCO 结果均已正确保存

---

## 九、项目特点

* 面向**开放指令**的遥感目标检测新范式
* 动态指令理解与任务分解机制
* 多任务统一、多模态协同
* 兼顾推理效率与语义泛化能力


