# FlagOS-Robo User Guide

## Overview

FlagOS-Robo is an integrated training and inference framework for AI models used in robots (Embodied Intelligence), built upon [FlagOS](https://flagos.io), the unified and open-source AI system software stack for various AI chips.

FlagOS-Robo can be deployed across diverse scenarios, ranging from edge to cloud. Being portable across various chip models, it enables efficient training, inference, and deployment for both Vision Language Models (VLMs) and Vision Language Action (VLA) models. VLMs usually act as the brain for task planning, while VLA models act as the cerebellum to output actions for robot control.

### Key Features

- **[FlagScale](https://github.com/flagos-ai/FlagScale)** as the user entrypoint, supporting robot-related AI model training and inference
- **Full Model Lifecycle**: Data loading, supervised fine-tuning (SFT), inference deployment, and evaluation
- **Multi-Format Data Loading**: Supports webdataset, Megatron-Energon, and lerobot dataset formats
- **[RoboOS](https://github.com/FlagOpen/RoboOS) Integration**: Cross-embodiment collaboration with different data formats and edge-cloud coordination
- **[RoboXStudio](https://ei2data.baai.ac.cn/home) Integration**: One-stop services including data collection, annotation, model training, simulation evaluation, and deployment

---

## Getting Started

### Supported Models

| Model | Type | Checkpoint | Train | Inference | Serve | Evaluate |
|-------|------|------------|-------|-----------|-------|----------|
| PI0 | VLA | [HuggingFace](https://huggingface.co/lerobot/pi0_base) | Yes | Yes | Yes | - |
| PI0.5 | VLA | [HuggingFace](https://huggingface.co/lerobot/pi05_libero_base) | Yes | Yes | Yes | - |
| RoboBrain-2.0 | VLM | [HuggingFace](https://huggingface.co/BAAI/RoboBrain2.0-7B) | Yes | Yes | Yes | Yes |
| RoboBrain-2.5 | VLM | [HuggingFace](https://huggingface.co/collections/BAAI/robobrain25) | Yes | Yes | Yes | Yes |
| RoboBrain-X0 | VLA | [HuggingFace](https://huggingface.co/BAAI/RoboBrain-X0-Preview) | Yes | - | Yes | - |
| Qwen-GR00T | VLA | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Yes | Yes | Yes | Yes |
| GR00T-N1.5 | VLA | [HuggingFace](https://huggingface.co/nvidia/GR00T-N1.5-3B) | Yes | - | Yes | - |

For detailed guides on each model, refer to the [FlagScale examples](https://github.com/flagos-ai/FlagScale/tree/main/examples).

---

## Evaluation

### FlagEval-Robo Platform

FlagOS-Robo integrates with the [FlagEval-Robo](https://embodiedverse.flageval.net/home) platform for testing and evaluation of embodied intelligence models.

### Auto-Evaluation Tool

Auto-Evaluation is a multi-chip adaptive model automatic evaluation tool built on the [FlagEval](https://flageval.baai.ac.cn/#/home) platform. It supports Embodied VLM models and is bound with ten classic datasets. Currently, it supports online evaluation only.

**Tool API Endpoint**: `120.92.17.239:5050`

#### Available APIs

| API | Method | Description |
|-----|--------|-------------|
| `/evaluation` | POST | Start evaluation |
| `/evaldiffs` | GET | Query evaluation results |
| `/stop_evaluation` | POST | Stop evaluation |
| `/resume_evaluation` | POST | Resume evaluation from breakpoint |
| `/evaluation_progress` | POST | View evaluation progress |
| `/evaluation_diffs` | POST | Compare results between multiple models |

#### Start an Evaluation

```bash
curl -X POST http://120.92.17.239:5050/evaluation \
  -H "Content-Type: application/json" \
  -d '{
    "eval_infos": [{
      "eval_model": "my-model-eval",
      "model": "Qwen/Qwen3-8B",
      "eval_url": "http://<your-host>:9010/v1/chat/completions",
      "tokenizer": "Qwen/Qwen3-8B"
    }],
    "domain": "MM",
    "mode": "EmbodiedVerse"
  }'
```

Key parameters:

- `eval_model`: Unique name for the evaluation task
- `model`: The deployed model name
- `eval_url`: Your model's evaluation endpoint
- `tokenizer`: Vendor and model information (e.g., `Qwen/Qwen3-8B`)
- `chip`: Chip used for evaluation (default: `Nvidia-H100`)

---

## RoboXStudio Platform

The [RoboXStudio](https://ei2data.baai.ac.cn/home) platform provides a SaaS environment for the full embodied intelligence pipeline:

- **Data Collection**: Real-robot data acquisition across diverse robotic embodiments
- **Data Annotation**: Labeling and data augmentation tools
- **Model Training**: SFT built on FlagOS with multi-chip adaptation
- **Simulation Evaluation**: Integrated testing and evaluation
- **Model Deployment**: End-to-end deployment pipeline

RoboXStudio has completed the full training and inference workflow based on FlagOS + BAAI RoboBrain-X0 model, achieving an end-to-end pipeline from model development and real-robot validation to task execution.

**Links**:

- Official Website: [https://ei2data.baai.ac.cn](https://ei2data.baai.ac.cn)
- Product Manual: [Feishu Wiki](https://jwolpxeehx.feishu.cn/wiki/GzbCwlYWwiqHvTk9b4icob3inug)
- Getting Started Guide: [Feishu Wiki](https://jwolpxeehx.feishu.cn/wiki/SbGzwjgakiQbHMkc0aecJbUunhg)
