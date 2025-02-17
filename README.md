# nanoverl

Run RL $\times$ LM experiments using minimal monkey patches for [verl](https://github.com/volcengine/verl). So that you do not need to modify the original code of verl, and keep up with the latest version of verl. We also do not use submodule to avoid the complexity of version control.

## Usage

First follow instructions in verl to install the main repo, then locally install this repo.
```bash
git clone https://github.com/koalazf99/nanoverl.git nanoverl
cd nanoverl
pip install -e .
```

## Examples

All scripts for RL experiments are in `nanoverl/example/`. For example, we can run the following script to train [deepscaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) dataset using [R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) with GRPO algorithm:

```bash
cd examples/deepscaler
python prepare_dataset.py
bash train_grpo_r1_distill_1b_8k.bash
```

The evaluation script is also a "nano" version thanks to [sglang](https://github.com/sgl-project/sglang). We use sglang-router to serve multiple backends.
```bash
python -m sglang_router.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --port 30000 --dp-size 8
python reasoning_eval.py \
    --data-path nanoverl/aime \
    --parallel 256 \
    --num-tries 16
```


## Local Installable Package Configuration
```bash
pip install poetry
poetry init
poetry build
```
