#!/bin/bash
#SBATCH --job-name=nanoverl_deepscaler
#SBATCH --partition=fan
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=100
#SBATCH --mem=512G
#SBATCH --output=./logs/slurm-%j.log
#SBATCH --error=./logs/slurm-%j.log
#SBATCH --exclusive
#SBATCH --time=12:00:00

# set -x

sleep 10
export worker_num=$SLURM_NNODES
JOBLOG="./logs/slurm-$SLURM_JOB_ID.log"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
export head_node=${nodes[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export port=30310
export address_head=$head_node_ip:$port


export VLLM_ATTENTION_BACKEND=XFORMERS
export EXPERIMENT_NAME=deepscaler-1.5b-8k
export OUTPUT_DIR=$HOME/nanoverl/checkpoints/${EXPERIMENT_NAME}
export GLOO_SOCKET_IFNAME=ens10f0np0


srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster
srun --nodes=1 --ntasks=1 -w "$head_node" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-gpus 8 --block & >> ${JOBLOG}

sleep 10

for ((i = 1; i < worker_num; i++)); do
    node_i=${nodes[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL,VLLM_ATTENTION_BACKEND=XFORMERS \
        ray start --address "$address_head" \
        --num-gpus 8 --block & >> ${JOBLOG}
    sleep 10
done

export PYTHONPATH=$PYTHONPATH:$HOME/nanoverl/
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
EXP_NAME="deepscaler-1.5b-8k"


python -m main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/nanoverl/data/parquet_data/deepscaler/train.parquet \
    data.val_files=$HOME/nanoverl/data/parquet_data/deepscaler/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_before_train=False \
    trainer.project_name='nanoverl' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$HOME/nanoverl/checkpoints/$EXP_NAME \
    trainer.total_epochs=30 >> ${JOBLOG}
