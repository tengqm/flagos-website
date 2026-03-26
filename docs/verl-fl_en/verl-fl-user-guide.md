# verl-FL User Guide

## Overview

verl-FL is a fork of [verl](https://github.com/volcengine/verl) designed to support diverse AI accelerators. It is built on top of [FlagOS](https://github.com/flagos-ai), a unified open-source AI system software stack, and integrates key components including the training engine [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL), [Transformer-Engine-FL](https://github.com/flagos-ai/TransformerEngine-FL), and the inference engine [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL).

verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient, and production-ready RL training framework for large language models (LLMs).

### Key Features

- **Diverse RL Algorithms**: PPO, GRPO, DAPO, DrGRPO, GMPO, SPPO, SPIN, RLOO, ReMax, REINFORCE++, PRIME, and more
- **Multi-Backend Training**: FSDP, FSDP2, and Megatron-LM (via Megatron-LM-FL) for training; vLLM, SGLang, and HF Transformers for inference/rollout
- **Multi-Hardware Support**: NVIDIA (CUDA), AMD (ROCm), Huawei Ascend (NPU) via platform abstraction
- **Scalable Architecture**: Single-controller design with Ray for orchestration, scaling from single-GPU to thousands of GPUs
- **Advanced Features**: Multi-turn tool calling, VLM RL, sequence packing, LoRA RL, expert parallelism, async training (fully async / one-step off-policy), speculative decoding for RL
- **Model Support**: HuggingFace models including Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek (up to 671B), and VLMs

---

## Getting Started

### Requirements

- Python >= 3.10
- CUDA >= 12.8

### Docker Installation (Recommended)

```bash
docker pull verlai/verl:latest
```

### pip Installation

```bash
# Create conda environment
conda create -n verl python=3.10
conda activate verl

# Install PyTorch with CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Clone and install verl-FL
git clone https://github.com/flagos-ai/verl-FL.git
cd verl-FL
pip install -e .

# Install vLLM for rollout
pip install vllm>=0.8.5

# Install Flash Attention
pip install flash-attn
```

### Quick Installation Script

```bash
# Installs vLLM, SGLang, and Megatron-Core backends
bash scripts/install_vllm_sglang_mcore.sh
```

### Training Backends

| Backend | Use Case | Installation |
|---------|----------|-------------|
| **FSDP** | Default, easy setup | Included with PyTorch |
| **FSDP2** | Latest PyTorch distributed | Included with PyTorch |
| **Megatron-LM** | Large-scale training | Via Megatron-LM-FL |

### Rollout Backends

| Backend | Use Case | Installation |
|---------|----------|-------------|
| **vLLM** | High-throughput inference | `pip install vllm>=0.8.5` |
| **SGLang** | Multi-turn, tool calling | `pip install sglang==0.5.6` |
| **HF Transformers** | Simple, no extra deps | Included |

---

## Installation

### Custom Environment Setup

```bash
# Create environment
conda create -n verl python=3.10
conda activate verl

# Install CUDA 12.8
conda install -c nvidia cuda-toolkit=12.8

# Install cuDNN
pip install nvidia-cudnn-cu12==9.10.1.2

# Install PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install verl-FL
cd verl-FL
pip install -e .

# Install Apex (optional, for Megatron backend)
pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation --config-settings="--build-option=--cpp_ext" \
    --config-settings="--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex
```

### AMD ROCm Support

verl-FL supports AMD GPUs via ROCm. See the `docs/amd_tutorial/` directory in the repository for detailed setup instructions.

### Ascend NPU Support

verl-FL supports Huawei Ascend NPUs. Install NPU-specific dependencies:

```bash
pip install -r requirements-npu.txt
```

See `docs/ascend_tutorial/` in the repository for detailed instructions.

---

## RL Algorithms

### PPO (Proximal Policy Optimization)

The standard RL algorithm for LLM post-training with actor-critic architecture and GAE (Generalized Advantage Estimation).

Key configuration:

```yaml
algorithm:
  kl_ctrl:
    type: fixed        # KL divergence control
    kl_coef: 0.001
trainer:
  train_batch_size: 256
  ppo_mini_batch_size: 64
  ppo_epochs: 1
```

### GRPO (Group Relative Policy Optimization)

Critic-free algorithm that estimates baselines from group scores instead of a learned value function.

Key configuration:

```yaml
rollout:
  n: 8                    # Number of samples per prompt
trainer:
  train_batch_size: 256
  ppo_mini_batch_size: 64
algorithm:
  loss_agg_mode: token     # or "seq" for sequence-level
```

### DAPO (Decoupled Alignment Policy Optimization)

Extension of GRPO with separated clip epsilons, dynamic sampling, and overlong reward shaping.

### Other Algorithms

- **GMPO**: Geometric-Mean Policy Optimization for stable training
- **SPPO**: Self-Play Preference Optimization
- **SPIN**: Self-Play Fine-Tuning with online DPO loss
- **DrGRPO**: GRPO with variance reduction

---

## Platform Abstraction

verl-FL includes a platform abstraction layer (`verl/plugin/platform/`) that provides a hardware-agnostic interface for multi-accelerator support.

### Supported Platforms

| Platform | Device | Status |
|----------|--------|--------|
| CUDA | NVIDIA GPUs | Full support |
| NPU | Huawei Ascend | Full support |
| CPU | CPU | Basic support |

### Adding a New Accelerator

To add support for a new accelerator (e.g., XPU, ROCm, MLU), implement the platform interface in `verl/plugin/platform/`. See the existing platform implementations as reference.

---

## Dataset Format

verl-FL uses the following RLHF dataset schema:

```json
{
  "data_source": "dataset_name",
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "4"
  }
}
```

Fields:

- `data_source`: Dataset identifier
- `prompt`: Chat-format messages (system/user/assistant roles)
- `ability`: Task category tag
- `reward_model`: Reward configuration (rule-based or model-based)
