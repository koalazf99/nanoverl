# nanoverl

Run RL $\times$ LM experiments using minimal monkey patches for [verl](https://github.com/volcengine/verl). So that you do not need to modify the original code of verl, and keep up with the latest version of verl. We also do not use submodule to avoid the complexity of version control.

## Usage

local install this repo as a package
```bash
git clone https://github.com/koalazf99/nanoverl.git nanoverl
cd nanoverl
pip install -e .
```

## Examples

All scripts for RL experiments are in `nanoverl/example/`. For example, we can run the following script to train [deepscaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) dataset using [R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) with GRPO algorithm:

```bash
# data prepare


```



## Local Installable Package Configuration
```bash
pip install poetry
poetry init
poetry build
```
