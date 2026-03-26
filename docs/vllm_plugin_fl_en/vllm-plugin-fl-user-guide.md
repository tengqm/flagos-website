# vllm-plugin-FL User Guide

## Overview

vllm-plugin-FL is a plugin for the [vLLM](https://github.com/vllm-project/vllm) inference/serving framework, built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends vLLM's capabilities and performance across diverse hardware environments.

Without changing vLLM's original interfaces or usage patterns, the same command can run model inference/serving on different chips.

### Supported Models

| Model | Status |
|-------|--------|
| Qwen3.5-397B-A17B | Supported |
| Qwen3-Next-80B-A3B | Supported |
| Qwen3-4B | Supported |
| MiniCPM-o 4.5 | Supported |
| GLM-5 | Supported |
| Qwen3.5-35B-A3B | Supported |

In theory, vllm-plugin-FL can support all models available in vLLM, as long as no unsupported operators are involved.

### Supported Chips

| Chip Vendor | Status |
|-------------|--------|
| NVIDIA | Supported |
| Ascend | Supported |
| Pingtouge-Zhenwu | Supported |
| Iluvatar | Supported |
| MetaX | Merging |
| Tsingmicro | Merging |
| Moore Threads | Merging |
| Hygon | Merging |

---

## Getting Started

### Installation

**Step 1**: Install vLLM v0.13.0:

```bash
# From official release
pip install vllm==0.13.0
# Or from the fork
pip install git+https://github.com/flagos-ai/vllm-FL.git
```

**Step 2**: Install vllm-plugin-FL:

```bash
git clone https://github.com/flagos-ai/vllm-plugin-FL
cd vllm-plugin-FL
pip install --no-build-isolation .
# Or editable install
pip install --no-build-isolation -e .
```

**Step 3**: Install [FlagGems](https://github.com/flagos-ai/FlagGems):

```bash
pip install -U scikit-build-core==0.11 pybind11 ninja cmake
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems
pip install --no-build-isolation .
```

**Step 4** (Optional): Install [FlagCX](https://github.com/flagos-ai/FlagCX) for multi-chip communication:

```bash
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX
git checkout -b v0.9.0
git submodule update --init --recursive
make USE_NVIDIA=1

export FLAGCX_PATH="$PWD"
cd plugin/torch/
FLAGCX_ADAPTOR=nvidia pip install . --no-build-isolation
```

```{note}
Set `FLAGCX_ADAPTOR` according to the current platform: `nvidia`, `ascend`, etc.
```

If multiple vLLM plugins are installed, specify vllm-plugin-FL via:

```bash
export VLLM_PLUGINS='fl'
```

### Additional Steps for Ascend

1. Install FlagTree:

```bash
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple --trusted-host=https://resource.flagos.net"
python3 -m pip install flagtree==0.4.0+ascend3.2 $RES
```

2. Set environment variable:

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

3. Enable eager execution — add `enforce_eager=True` to the `LLM` constructor or pass `--enforce-eager` on the command line.

---

## Quick Start

### Offline Batched Inference

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]

sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
llm = LLM(model="Qwen/Qwen3-4B", max_num_batched_tokens=16384, max_num_seqs=2048)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Using Native CUDA Operators

To bypass FlagGems and use original CUDA operators:

```bash
export USE_FLAGGEMS=0
```

### Using CUDA Communication Library

To bypass FlagCX and use the original CUDA communication:

```bash
unset FLAGCX_PATH
```

---

## Dispatch Mechanism

vllm-plugin-FL includes a flexible operator dispatch system that selects between different backend implementations (FlagGems, PyTorch, vendor-specific) based on availability and policy.

### Backend Priority

| Kind | Priority | Description |
|------|----------|-------------|
| DEFAULT (FlagGems) | 150 | FlagOS default implementations |
| VENDOR | 100 | Vendor-specific implementations (CUDA, Ascend) |
| REFERENCE | 50 | PyTorch native implementations |

### Dispatch Flow

1. Check cache for previously resolved operator
2. Get all registered implementations from registry
3. Filter by vendor allow/deny lists
4. Check implementation availability
5. Sort by priority and selection order
6. Cache and return selected implementation

### Supported Operators

| Operator | Description | FlagGems | Reference | Vendor |
|----------|-------------|----------|-----------|--------|
| `silu_and_mul` | SiLU activation + element-wise multiplication | Yes | Yes | Yes |
| `rms_norm` | RMS normalization | Yes | Yes | Yes |
| `rotary_embedding` | Rotary position embedding | Yes | Yes | Yes |
| `attention_backend` | Attention backend class path | Yes | - | Yes |

---

## Configuration

### Environment Variables

#### Core Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_PREFER_ENABLED` | `true` | Global switch for dispatch features |
| `VLLM_FL_CONFIG` | (none) | Path to YAML config file (complete override) |
| `VLLM_FL_PLATFORM` | (auto) | Force platform: `ascend`, `cuda` |

#### Backend Selection

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_PREFER` | `flagos` | Preferred backend: `flagos`, `vendor`, `reference` |
| `VLLM_FL_STRICT` | `0` | Strict mode: `1` = fail on error, `0` = try fallback |
| `VLLM_FL_PER_OP` | (none) | Per-operator order: `op1=a\|b\|c;op2=x\|y` |
| `VLLM_FL_ALLOW_VENDORS` | (none) | Vendor whitelist, comma-separated |
| `VLLM_FL_DENY_VENDORS` | (none) | Vendor blacklist, comma-separated |

#### FlagGems Control

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FLAGGEMS` | `true` | Enable/disable FlagGems |
| `VLLM_FL_FLAGOS_WHITELIST` | (none) | FlagGems ops whitelist |
| `VLLM_FL_FLAGOS_BLACKLIST` | (none) | FlagGems ops blacklist |

#### Debug & Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_FL_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `VLLM_FL_DISPATCH_DEBUG` | `0` | Enable dispatch debug mode |

### YAML Configuration

Create a custom config file and set `VLLM_FL_CONFIG`:

```bash
export VLLM_FL_CONFIG=/path/to/vllm_fl_dispatch.yaml
```

Example configuration:

```yaml
# Preferred backend type
prefer: vendor
# Strict mode
strict: false
# Vendor whitelist
allow_vendors:
  - cuda
# Per-operator backend selection
op_backends:
  rms_norm:
    - vendor
    - flagos
  silu_and_mul:
    - vendor:cuda
    - flagos
    - reference
# FlagGems operator blacklist
flagos_blacklist:
  - to_copy
  - zeros
  - mm
```

### Configuration Priority

1. `VLLM_FL_CONFIG` (user config file) — complete override
2. Environment variables — override specific items
3. Platform config file (`ascend.yaml` / `cuda.yaml`) — auto-detected defaults
4. Built-in default values

---

## Adding Vendor Backends

### Built-in Vendor Backend

Create the following directory structure:

```
backends/vendor/<vendor_name>/
├── __init__.py
├── <vendor_name>.py        # Backend class
├── register_ops.py         # Registration function
└── impl/                   # Operator implementations
    ├── activation.py
    ├── normalization.py
    └── rotary.py
```

The backend class must inherit from `Backend` and implement `is_available()` along with operator methods.

### External Plugin Package

Create a separate package with setuptools entry points:

```python
# setup.py
setup(
    name="vllm-plugin-<vendor>",
    entry_points={
        "vllm_fl.plugin": [
            "<vendor> = vllm_fl_<vendor>.register_ops:register_builtins",
        ],
    },
)
```

### Environment-based Plugin

```bash
export VLLM_FL_PLUGIN_MODULES=my_custom_backend.register_ops
```
