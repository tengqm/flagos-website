# User Guide

This guide covers how to configure FlagScale and run training, inference, serving, and reinforcement learning tasks.

## Step 1: Configure YAML files

FlagScale uses [Hydra](https://hydra.cc/) for configuration management. Every task is driven by two YAML files that work together: an experiment-level file and a task-level file, both in the `examples/` directory. Before running the task, you need to configure these files first.

### Experiment-level YAML

Use the `examples/qwen3/conf/serve.yaml` as an example to explain this configuration file.

The experiment-level file is the entry point for `flagscale` commands. It defines a global context for the run:

* where outputs are stored: `exp_dir: outputs/${experiment.exp_name}`

* which backend engine to use: `backend: vllm`

* which task-level file to load: `defaults: - serve: 8b`

### Task-level YAML

Use the `examples/qwen3/conf/serve/8b.yaml` as an example to explain this configuration file.

The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference. Every parameter in this file maps directly to an argument accepted by the backend engine, with hyphens (-) replaced by underscores (\_).

## Step 2： Run tasks

FlagScale provides a unified runner for various tasks, including training, inference, reinforcement learning, and serving. Simply specify the configuration file to run the task with a single `flagscale` command. The runner will automatically load the configurations and execute the task. The following sections demonstrate how to run a distributed training task.

### Train

Require Megatron-LM-FL enviroment

1. Prepare dataset demo and tokenizer:

   * Download dataset: We provide a small processed data ([<u>bin</u>](https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.bin) and [<u>idx</u>](https://baai-flagscale.ks3-cn-beijing.ksyuncs.com/datasets/enron_emails_demo_text_document_qwen/enron_emails_demo_text_document_qwen.idx)) from the [<u>Pile</u>](https://pile.eleuther.ai/) dataset.

   * Download tokenizer

2. Modify the paths of the dataset and tokenizer in the task-level YAML file and the model name in the experiment-level YAML file

   1. **Task-level YAML file**: Modify the `data_path` and `tokenizer_path` in ./examples/qwen3/conf/train/0\_6b.yaml.

   2. **Experiment-level YAML**: Modify `train `model name  in ./examples/qwen3/conf/train.yaml. The value must match the file name **0\_6b**.yaml above.

3. Start the distributed training job:

4. Stop the distributed training job:

### Inference

Require vLLM-FL environment

1. Download inference model

2. Modify the model path in the task-level YAML file and the model name in the experiment-level YAML file

   1. **Task-level YAML file**: Modify the `model` path in `./examples/qwen3/conf/inference/4b.yaml`.

   2. **Experiment-level YAML**: Modify `inference`model name in `./examples/qwen3/conf/inference_fl.yaml`. The value must match the file name **4b**.yaml above.

3. Start inference:

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