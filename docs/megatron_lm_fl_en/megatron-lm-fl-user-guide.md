# Megatron-LM-FL User Guide

## Overview

Megatron-LM-FL is a fork of [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) that introduces a **plugin-based architecture** for supporting diverse AI chips, built on top of [FlagOS](https://github.com/flagos-ai), a unified open-source AI system software stack.

While upstream Megatron-LM is optimized exclusively for NVIDIA GPUs, Megatron-LM-FL extends it with a hardware abstraction layer that enables training on multiple platforms — including CUDA, MUSA, TXDA, and CPU — with minimal code intrusion to the core library.

### Key Features

- **Plugin System**: `@overridable` / `@override` decorator mechanism allows platform-specific implementations to replace core methods without modifying upstream code
- **Multi-Platform Support**: Hardware abstraction via `PlatformBase` with implementations for CUDA, MUSA (Moore Threads), TXDA (Tsingmicro), and CPU
- **Full Upstream Compatibility**: All upstream Megatron-LM features are preserved, including advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision (FP16, BF16, FP8), and GPU-optimized kernels

### Megatron Core

**Megatron Core** (`megatron.core`) is the production-ready PyTorch-based library at the heart of Megatron-LM. It provides GPU-optimized techniques and system-level optimizations abstracted into composable and modular APIs:

- **Transformer Components**: Attention mechanisms, MLP layers, embeddings
- **Memory Management**: Activation recomputation
- **FP8 Precision**: Optimized for NVIDIA Hopper, Ada, and Blackwell GPUs
- **Parallelism**: Tensor Parallelism (TP), Pipeline Parallelism (PP), Context Parallelism (CP), Expert Parallelism (EP)
- **Model Architectures**: GPT, LLaMA, Qwen, Mixtral, BERT, T5, Mamba, multimodal models, and more

### Project Structure

```
Megatron-LM-FL/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── plugin/                  # FL plugin system (multi-chip support)
│   │   ├── platform/            # Hardware platform abstraction
│   │   ├── distributed/         # Distributed training overrides
│   │   ├── optimizer/           # Optimizer overrides
│   │   └── decorators.py        # @overridable / @override mechanism
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   └── post_training/           # Post-training (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

---

## Getting Started

### Quick Start

```bash
# Clone Megatron-LM-FL repository (includes megatron.core and megatron.plugin)
git clone https://github.com/flagos-ai/Megatron-LM-FL.git
cd Megatron-LM-FL
# Install Megatron-LM-FL with required dependencies
pip install --no-build-isolation .[mlm,dev]
```

### Simple Training Example

```bash
# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

### LLama-3 Training Example

```bash
# 8 GPUs, FP8 precision, mock data
./examples/llama/train_llama3_8b_fp8.sh
```

---

## Installation

### Docker (Recommended)

We recommend using [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) releases for optimal compatibility. This container comes with all dependencies pre-installed:

- PyTorch (latest stable version)
- CUDA, cuDNN, NCCL (latest stable versions)
- FP8 support on NVIDIA Hopper, Ada, and Blackwell GPUs

```bash
# Run container with mounted directories
docker run --runtime --nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.04-py3
```

### Pip Installation

Megatron Core offers two dependency profiles:

- `dev`: Moving head that supports the most recent upstream dependencies
- `lts`: Long-term support of NGC PyTorch 24.01

Both can be combined with `mlm` which adds Megatron-LM dependencies on top of Megatron Core.

```bash
# Install the latest release dependencies
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,dev]
```

```bash
# Install packages for LTS support
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation megatron-core[mlm,lts]
```

For a minimal version with only torch:

```bash
pip install megatron-core
```

### Source Installation (Megatron-LM-FL)

```bash
git clone https://github.com/flagos-ai/Megatron-LM-FL.git
cd Megatron-LM-FL
pip install --no-build-isolation .[mlm,dev]
```

### System Requirements

**Hardware:**

- FP8 Support: NVIDIA Hopper, Ada, Blackwell GPUs
- Recommended: NVIDIA Turing architecture or later

**Software:**

- Python >= 3.10 (3.12 recommended)
- PyTorch: Latest stable version
- CUDA/cuDNN/NCCL: Latest stable versions
- Transformer Engine: Latest stable version

---

## Platform Support

Megatron-LM-FL supports multiple hardware platforms through the `megatron.plugin.platform` abstraction layer.

### Supported Platforms

| Platform | Device | Description |
|----------|--------|-------------|
| **CUDA** | NVIDIA GPUs | Full feature support (default) |
| **MUSA** | Moore Threads GPUs | Moore Threads MUSA platform |
| **TXDA** | Tsingmicro GPUs | Tsingmicro TXDA platform |
| **CPU** | CPU | CPU-only fallback |

### Platform Selection

The platform is selected automatically in priority order: CUDA > MUSA > TXDA > CPU. You can override this with the `MG_PLATFORM` environment variable:

```bash
export MG_PLATFORM=cuda   # Force CUDA platform
export MG_PLATFORM=musa   # Force MUSA platform
export MG_PLATFORM=txda   # Force TXDA platform
export MG_PLATFORM=cpu    # Force CPU platform
```

### Plugin System

The plugin system enables platform-specific overrides through two decorators:

**`@overridable`** — Marks a method or function in `megatron.core` as replaceable by a plugin:

```python
# In megatron/core/some_module.py
from megatron.plugin.decorators import overridable

@overridable
def some_function(...):
    # Original implementation (fallback if no plugin)
    ...
```

**`@override`** — Registers a plugin implementation that replaces an `@overridable` target:

```python
# In megatron/plugin/some_module.py
from megatron.plugin.decorators import override

@override("some_module", "some_function")
def some_function(...):
    # Platform-specific implementation
    ...
```

The plugin system automatically maps `megatron.core.X.Y` to `megatron.plugin.X.Y` and caches lookup results for performance.

---

## Training

### Data Preparation

Training data uses JSONL format:

```json
{"text": "Your training text here..."}
{"text": "Another training sample..."}
```

Preprocess data with:

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

Key arguments:

- `--input`: Path to input JSON/JSONL file
- `--output-prefix`: Prefix for output binary files (.bin and .idx)
- `--tokenizer-type`: Tokenizer type (`HuggingFaceTokenizer`, `GPT2BPETokenizer`, etc.)
- `--tokenizer-model`: Path to tokenizer model file
- `--workers`: Number of parallel workers for processing
- `--append-eod`: Add end-of-document token

---

## Parallelism Strategies

### Data Parallelism (DP)

```bash
# Standard DDP - replicate model on each GPU
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --data-parallel-sharding-strategy no_shard

# Megatron's optimized FSDP (~15% faster than PyTorch FSDP2)
--use-custom-fsdp

# PyTorch FSDP2
--use-torch-fsdp2

# Sharding strategies
--data-parallel-sharding-strategy optim              # ZeRO-1
--data-parallel-sharding-strategy optim_grads        # ZeRO-2
--data-parallel-sharding-strategy optim_grads_params # ZeRO-3
```

### Tensor Parallelism (TP)

Split individual model layers across GPUs:

```bash
--tensor-model-parallel-size 4  # 4-way tensor parallelism
--sequence-parallel             # Enable sequence parallelism (recommended with TP)
```

### Pipeline Parallelism (PP)

Split model depth across GPUs:

```bash
--pipeline-model-parallel-size 8          # 8 pipeline stages
--virtual-pipeline-model-parallel-size 4  # Virtual pipeline for better load balancing
```

### Context Parallelism (CP)

Split long sequences across GPUs:

```bash
--context-parallel-size 2                    # 2-way context parallelism
--cp-comm-type p2p                          # Communication: p2p, a2a, allgather, a2a+p2p
--hierarchical-context-parallel-sizes 2 4   # Hierarchical context parallelism
```

### Expert Parallelism (EP)

For Mixture of Experts (MoE) models:

```bash
--expert-model-parallel-size 4  # 4-way expert parallelism
--num-experts 8                 # 8 experts per MoE layer
--moe-grouped-gemm              # Optimize expert computation
```

### Parallelism Selection Guide

| Model | Size | GPUs | TP | PP | CP | EP | Notes |
|-------|------|------|----|----|----|----|-------|
| **LLama-3** | 8B | 8 | 1 | 1 | 2 | 1 | CP for long seqlen (8K) |
| **LLama-3** | 70B | 64 | 4 | 4 | 2 | 1 | TP+PP |
| **LLama-3.1** | 405B | 1024 | 8 | 8 | 2 | 1 | 3D parallelism for scale |
| **GPT-3** | 175B | 128-512 | 4 | 8 | 1 | 1 | Large model config |
| **Mixtral** | 8x7B | 64 | 1 | 4 | 1 | 8 | EP for MoE |
| **Mixtral** | 8x22B | 256 | 4 | 4 | 8 | 8 | Combined TP+EP for large MoE |
| **DeepSeek-V3** | 671B | 1024 | 2 | 16 | 1 | 64 | Large MoE config |

```{note}
When combining Expert Parallelism (EP) with Tensor Parallelism (TP), **Sequence Parallelism (SP) must be enabled**.
```

---

## Performance Optimizations

| Feature | Flag | Benefit |
|---------|------|---------|
| **FlashAttention** | `--attention-backend` | Faster attention and lower memory usage |
| **FP8 Training** | `--fp8-hybrid` | Faster training |
| **Activation Checkpointing** | `--recompute-activations` | Reduced memory usage |
| **DP Communication Overlap** | `--overlap-grad-reduce` | Faster distributed training |
| **Distributed Optimizer** | `--use-distributed-optimizer` | Reduced checkpointing time |

### Mixed Precision Training

```bash
--fp16                    # Standard FP16
--bf16                    # BFloat16 (recommended for large models)
--fp8-hybrid              # FP8 training (Hopper, Ada, and Blackwell GPUs)
```

### Activation Checkpointing

```bash
# For limited memory
--recompute-activations

# For extreme memory constraints
--recompute-granularity full \
--recompute-method uniform
```

### Communication Overlap

```bash
--overlap-grad-reduce
--overlap-param-gather
```

### Distributed Optimizer

```bash
--use-distributed-optimizer
```
