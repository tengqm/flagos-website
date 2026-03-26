# Install FlagScale

Read [Requirements](requirements.md) before proceeding.

## Steps

1. Install FlagScale:

   1. From source

      ```{code-block} python
      git clone https://github.com/flagos-ai/FlagScale.git
      cd FlagScale
      pip install . --verbose
      ```

   1. Via pip.

      ```{code-block} python
      pip install flagscale --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
      ```

2. Install backends as needed:

   - Inference and serving backend：

    Install vLLM-plugin-FL.

    ```{code-block} shell
    pip install vllm==0.13.0
    pip install vllm-plugin-fl==0.1.0+vllm0.13.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
    ```

    You may need to install other FlagOS components. For more information, see [<u>vLLM-plugin-FL</u>](https://github.com/flagos-ai/vllm-plugin-FL).

   - Training backend:

     Install Megatron-LM-FL.

     ```{code-block} shell
     pip install megatron_core==0.1.0+megatron0.15.0rc7 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
     pip install transformer_engine==0.1.0+te2.9.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
     ```

     For more information, see [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL).

     Install TransformerEngine-FL.

     ```{code-block} shell
     pip install transformer_engine==0.1.0+te2.9.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
     ```

   For more information, see [TransformerEngine-FL](https://github.com/flagos-ai/TransformerEngine-FL).  

   - Reinforce learning backend:

     Install verl-FL.

     ```{code-block} shell
     pip install verl==0.1.0+verl0.7.0 --extra-index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple
     ```

   To get full installation instructions, see [verl-FL](https://github.com/flagos-ai/verl-FL.git).

3. Verify FlagScale installation

```{code-block} shell
pip show flagscale
```
