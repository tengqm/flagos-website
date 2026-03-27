# TransformerEngine-FL User Guide

## Overview

TransformerEngine-FL is a fork of [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) that introduces a **plugin-based architecture** for supporting diverse AI chips, built on top of [FlagOS](https://github.com/flagos-ai), a unified open-source AI system software stack.

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper, Ada, and Blackwell GPUs, to provide better performance with lower memory utilization in both training and inference. TransformerEngine-FL extends this with a plugin system that enables non-NVIDIA backends to provide operator implementations.

### Key Features

- **FP8 Training & Inference**: Easy-to-use modules for building Transformer layers with FP8 support on NVIDIA Hopper, Ada, and Blackwell GPUs
- **Optimized Kernels**: Fused kernels for attention, normalization, activation, GEMM, and more
- **Multi-Precision Support**: FP8, FP16, BF16 across NVIDIA Ampere architecture and later
- **Plugin System** (FL-specific): Extensible operator dispatch with support for custom backends (in-tree and out-of-tree), enabling multi-chip support
- **Framework Support**: PyTorch and JAX integrations
- **Broad Integration**: Works with Megatron-LM, NeMo, DeepSpeed, HF Accelerate, Lightning, and more

---

## Getting Started

### Quick Example (PyTorch)

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Enable autocasting for the forward pass
with te.autocast(enabled=True, recipe=fp8_recipe):
    out = model(inp)

loss = out.sum()
loss.backward()
```

### Quick Example (JAX/Flax)

```python
import flax
import jax
import jax.numpy as jnp
import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.common import recipe

BATCH = 32
SEQLEN = 128
HIDDEN = 1024

rng = jax.random.PRNGKey(0)
init_rng, data_rng = jax.random.split(rng)
inp = jax.random.normal(data_rng, [BATCH, SEQLEN, HIDDEN], jnp.float32)

fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.HYBRID)

with te.autocast(enabled=True, recipe=fp8_recipe):
    model = te_flax.DenseGeneral(features=HIDDEN)
    variables = model.init(init_rng, inp)
```

---

## Installation

### System Requirements

- **Hardware**: NVIDIA Blackwell, Hopper, Grace Hopper, Ada, or Ampere GPUs
- **OS**: Linux (official), WSL2 (limited support)
- **CUDA**: 12.1+ (Hopper/Ada/Ampere), 12.8+ (Blackwell) with compatible NVIDIA drivers
- **cuDNN**: 9.3+
- **Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **Python**: 3.12 recommended
- **Source build**: CMake 3.18+, Ninja, Git 2.17+, pybind11 2.6.0+

```{note}
FP8 features require Compute Capability 8.9+ (Ada/Hopper/Blackwell).
```

### Docker (Recommended)

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.08-py3
docker run --gpus all -it --rm nvcr.io/nvidia/jax:25.08-py3
```

### pip Installation

```bash
# PyTorch
pip install --no-build-isolation transformer_engine[pytorch]

# JAX
pip install --no-build-isolation transformer_engine[jax]

# Both frameworks
pip install --no-build-isolation transformer_engine[pytorch,jax]
```

### conda Installation

```bash
conda install -c conda-forge transformer-engine-torch
```

### Source Installation

```bash
git clone https://github.com/flagos-ai/TransformerEngine-FL.git
cd TransformerEngine-FL
git submodule update --init --recursive
pip install --no-build-isolation .
```

---

## Plugin System (FL-specific)

TransformerEngine-FL adds a plugin-based operator dispatch system in `transformer_engine/plugin/`. It allows alternative backend implementations to be registered and selected at runtime, enabling multi-chip support without modifying the core library.

### Architecture

The plugin system consists of:

- **OpRegistry**: Thread-safe registry for operator implementations
- **OpManager**: Core dispatch manager that selects the best available backend
- **Policy**: Configurable backend selection policy
- **Discovery**: Plugin discovery via environment variables and setuptools entry points

### Backend Priority

| Kind | Priority | Description |
|------|----------|-------------|
| DEFAULT (FlagOS) | 150 | FlagGems-based implementations |
| VENDOR | 100 | Vendor-specific implementations |
| REFERENCE | 50 | PyTorch native implementations |

### In-Tree Approach

Register backends directly in the codebase:

```python
from transformer_engine.plugin.core import OpRegistry, OpManager, OpImpl

registry = OpRegistry()
registry.register(OpImpl(
    op_name="my_op",
    impl_id="vendor.my_vendor",
    kind=BackendImplKind.VENDOR,
    fn=my_implementation,
    vendor="my_vendor",
))
```

### Out-of-Tree Approach

Create a separate plugin package with a `register()` entry point, then load it via:

```bash
# Via environment variable
export TE_FL_PLUGIN_MODULES=my_plugin_module

# Or via pip install (auto-discovered via entry points)
pip install my-te-plugin
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TE_FL_PREFER` | `flagos` | Preferred backend: `flagos`, `vendor`, `reference` |
| `TE_FL_STRICT` | `0` | Strict mode: fail on error vs. try fallback |
| `TE_FL_ALLOW_VENDORS` | (none) | Vendor whitelist, comma-separated |
| `TE_FL_DENY_VENDORS` | (none) | Vendor blacklist, comma-separated |
| `TE_FL_PER_OP` | (none) | Per-operator backend order |
| `TE_FL_PLUGIN_MODULES` | (none) | External plugin modules, comma-separated |
| `TE_FL_SKIP_CUDA` | `0` | Set to `1` to skip CUDA compilation (FL-only mode) |
| `TEFL_LOG_LEVEL` | (none) | Logging level for plugin system |

---

## FP8 Training

Transformer Engine provides automatic mixed precision training with FP8. FP8 convergence has been validated across a range of models:

- T5 (770M)
- MPT (1.3B, 13B)
- GPT (5B, 22B, 175B)
- LLama2 (7B, 70B)
- T5 (11B)

### FP8 Formats

- **E4M3**: 4 exponent bits, 3 mantissa bits — used for forward pass
- **E5M2**: 5 exponent bits, 2 mantissa bits — used for backward pass
- **HYBRID**: Combines both formats automatically

### Usage

```python
from transformer_engine.common import recipe

# Delayed scaling recipe (recommended)
fp8_recipe = recipe.DelayedScaling(
    margin=0,
    fp8_format=recipe.Format.HYBRID
)

with te.autocast(enabled=True, recipe=fp8_recipe):
    output = model(input)
```

---

## Integrations

TransformerEngine integrates with the following frameworks and libraries:

- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** — Large-scale model training
- **[NeMo Framework](https://docs.nvidia.com/nemo-framework/)** — Enterprise AI framework
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** — Distributed training optimization
- **[HF Accelerate](https://github.com/huggingface/accelerate)** — Hugging Face distributed training
- **[Lightning](https://lightning.ai/)** — PyTorch Lightning
- **[MosaicML Composer](https://github.com/mosaicml/composer)** — ML training library
- **[Nanotron](https://github.com/huggingface/nanotron)** — Efficient LLM training
- **[Colossal-AI](https://github.com/hpcaitech/ColossalAI)** — Distributed training framework
