# User Guide

This guide covers how to configure FlagScale and run training, inference, serving, and reinforcement learning tasks.

## Step 1: Configure YAML files

FlagScale uses [Hydra](https://hydra.cc/) for configuration management. Every task is driven by two YAML files that work together: an experiment-level file and a task-level file, both in the `examples/` directory. Before running the task, you need to configure these files first.

### Experiment-level YAML

Use the `examples/qwen3/conf/serve.yaml` as an example to explain this configuration file.

The experiment-level file is the entry point for `flagscale` commands. It defines a global context for the run:

- where outputs are stored: `exp_dir: outputs/${experiment.exp_name}`

- which backend engine to use: `backend: vllm`

- which task-level file to load: `defaults: - serve: 8b`


```{code-block} yaml
# Example: examples/qwen3/conf/serve.yaml
defaults:
- _self_
- serve: 8b

experiment:
  exp_name: qwen3_8b
  exp_dir: outputs/${experiment.exp_name}
  task:
    type: serve
    backend: vllm
  runner:
    hostfile: null
    deploy:
      use_fs_serve: false
  envs:
    CUDA_VISIBLE_DEVICES: 0
    CUDA_DEVICE_MAX_CONNECTIONS: 1

action: run

hydra:
  run:
    dir: ${experiment.exp_dir}/hydra
```

### Task-level YAML

Use the `examples/qwen3/conf/serve/8b.yaml` as an example to explain this configuration file.

The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference. Every parameter in this file maps directly to an argument accepted by the backend engine, with hyphens (-) replaced by underscores (\_).

```{code-block} yaml
# Example: examples/qwen3/conf/serve/8b.yaml
- serve_id: vllm_model
  engine_args:
    model: ${oc.env:QWEN3_PATH}
    host: 0.0.0.0
    uvicorn_log_level: warning
    port: ${oc.env:QWEN3_PORT}
    gpu_memory_utilization: 0.9
    trust_remote_code: true
    no_enable_prefix_caching: true
    compilation_config: '{"full_cuda_graph": true}'
```


## Step 2： Run tasks

FlagScale provides a unified runner for various tasks, including training, inference, reinforcement learning, and serving. Simply specify the configuration file to run the task with a single `flagscale` command. The runner will automatically load the configurations and execute the task. The following sections demonstrate how to run a distributed training task.

### Train

Require Megatron-LM-FL enviroment

1. Prepare dataset demo and tokenizer:

   - Download dataset: We provide a small processed data ([bin](https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.bin) and [idx](https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.
  
   ```{code-block} shell
   mkdir -p ./data && cd ./data
   wget https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.idx
   wget https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.bin
   ```
   - Download tokenizer
  
   ```{code-block} shell
   mkdir -p ./qwentokenizer && cd ./qwentokenizer
   wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/tokenizer_config.json" -O tokenizer_config.json
   wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/qwen.tiktoken" -O qwen.tiktoken
   wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/qwen_generation_utils.py" -O qwen_generation_utils.py
   wget "https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/tokenizers/qwentokenizer/tokenization_qwen.py" -O tokenization_qwen.py
   ```
  
2. Modify the paths of the dataset and tokenizer in the task-level YAML file and the model name in the experiment-level YAML file

   1. **Task-level YAML file**: Modify the `data_path` and `tokenizer_path` in ./examples/qwen3/conf/train/0\_6b.yaml.

   ```{code-block} yaml
   data:
    data_path: ./data/enron_emails_demo_text_document_qwen    # modify data_path here
    split: 1
    no_mmap_bin_files: true
    tokenizer:
        legacy_tokenizer: true
        tokenizer_type: QwenTokenizerFS
        tokenizer_path: ./qwentokenizer   # modify tokenizer_path here
        vocab_size: 151936
        make_vocab_size_divisible_by: 64
   ```

   2. **Experiment-level YAML**: Modify `train `model name  in ./examples/qwen3/conf/train.yaml. The value must match the file name **0\_6b**.yaml above.
   
   ```{code-block} yaml
   data:
    data_path: ./data/enron_emails_demo_text_document_qwen    # modify data_path here
    split: 1
    no_mmap_bin_files: true
    tokenizer:
        legacy_tokenizer: true
        tokenizer_type: QwenTokenizerFS
        tokenizer_path: ./qwentokenizer   # modify tokenizer_path here
        vocab_size: 151936
        make_vocab_size_divisible_by: 64
   ```

3. Start the distributed training job:

   ```{code-block} shell
   flagscale train qwen3 --config ./examples/qwen3/conf/train.yaml
   # or
   flagscale train qwen3 -c ./examples/qwen3/conf/train.yaml
   ```

4. Stop the distributed training job:

   ```{code-block} shell
   flagscale train qwen3 --stop
   ```


### Inference

Require vLLM-FL environment

1. Download inference model

   ```{code-block} shell
   modelscope download --model Qwen/Qwen3-4B --local_dir ./Qwen3-4B
   ```

2. Modify the model path in the task-level YAML file and the model name in the experiment-level YAML file

   1. **Task-level YAML file**: Modify the `model` path in `./examples/qwen3/conf/inference/4b.yaml`.

   ```{code-block} python
   llm:
    model: ./Qwen3-4B         # modify: Set model directory
    trust_remote_code: true
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    gpu_memory_utilization: 0.9
    seed: 1234
   ```

   2. **Experiment-level YAML**: Modify `inference`model name in `./examples/qwen3/conf/inference_fl.yaml`. The value must match the file name **4b**.yaml above.

   ```{code-block} python
   defaults:
   - _self_
   - inference: 4b    # modify: Inference value must match its corresponding config file name
   ```

3. Start inference:

```{code-block} python
flagscale inference qwen3 --config ./examples/qwen3/conf/inference_fl.yaml
# or
flagscale inference qwen3 -c ./examples/qwen3/conf/inference_fl.yaml
```


### Serve

1. Download serving model

2. Modify the model path in the task-level YAML file and the model name in the experiment-level YAML file

   1. **Task-level YAML file**: Modify the `model` path in `./examples/qwen3/conf/serve/0_6b.yaml`.

   2. **Experiment-level YAML**: Modify `serve` model name in `./examples/qwen3/conf/serve.yaml`.

3. Start the server:

4. Stop the server:

### Reinforcement Learning

Require verl-FL environment

1. Download model

2. Download dataset

3. Modify the model path in the task-level YAML file and the model name in the experiment-level YAML file

   1. **Task-level YAML file**: In `./examples/qwen3/conf/rl/0_6b.yaml`

      * Modify the `train_files` (train dataset path) and `val_files` (test dataset path).

      * Modify the `path`(model checkpoint path).

   2. **Experiment-level YAML**: Modify exp\_dir (experiment directory) and runtime\_env (runtime environment path) in `./examples/qwen3/conf/rl.yaml`.

4. Start reinforcement learning:

You can check the output in your experiment directory.

5. Stop reinforcement learning:

   Or force to stop Ray cluster.