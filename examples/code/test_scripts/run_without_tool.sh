pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# apt-get install gawk -y
# cp -r /gfs/space/chatrl/users/wlw_temp/wlw/firejail ~
# cd ~/firejail
# while true; do
#    make clean
#    ./configure
#    make
#    make install-strip
#    firejail --version
#    ret=$?
#    if [ $ret -ne 0 ]; then
#        echo 'install firejail failed: $ret'
#    else
#        break
#    fi
# done

export REPO_PATH=/gfs/space/chatrl/users/wlw_temp/slime_code/slime/
cd ${REPO_PATH}

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export nnodes=1
export num_gpus_per_node=4
ray start --head \
  --node-ip-address ${MASTER_ADDR} \
  --num-gpus ${num_gpus_per_node} \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

export RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export TIMESTAMP=$(date +"%y%m%d%H%M%S")


export max_resp_len=$(( 1024 * 16 ))
export max_context_len=$(( 1024 * 16 ))
export rollout_batch_size=16
export rollout_n=8
export global_batch_size=128
export num_steps_per_rollout=1
export NCCL_GRAPH_REGISTER=0

# export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=True
# export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=1024

export EXP_NAME=for_sandbox_debug
export project_name=slime_30B-A3B_code_debug_single
export EXP_DIR=/gfs/space/chatrl/users/wlw_temp/slime_code
export DUMP_DIR=${EXP_DIR}/dump_details_${TIMESTAMP}
export LOG_FILE=${EXP_DIR}/logs/output_without_${TIMESTAMP}.log
export CKPT_SAVE_PATH=${EXP_DIR}/checkpoints_${TIMESTAMP}
export TENSORBOARD_DIR=${EXP_DIR}/tensorboard_log/qwen3-8B_withouttool_${TIMESTAMP}

# export MODEL_PATH=/gfs/space/chatrl/public/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
# export DIST_MODEL_PATH=/gfs/space/chatrl/public/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B-dist

# export MODEL_PATH=/gfs/space/chatrl/public/models/Qwen3-4B
# export DIST_MODEL_PATH=/gfs/space/chatrl/public/models/Qwen3-4Btorch_dist

export MODEL_PATH=/gfs/space/chatrl/public/models/Qwen3-8B
export DIST_MODEL_PATH=/gfs/space/chatrl/public/models/Qwen3-8B-dist

#!/bin/bash
set -ex
pwd

source "/gfs/space/chatrl/users/wlw_temp/slime_code/slime/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_PATH}
   #--hf-checkpoint /root/Qwen3-30B-A3B-FP8
   --ref-load ${DIST_MODEL_PATH}
   --load ${MODEL_PATH}
   #--save ${CKPT_SAVE_PATH}
   #--save-interval 30
)

ROLLOUT_ARGS=(
   #--rollout-function-path examples.code.custom_multi_turn.generate_rollout
   #--prompt-data /gfs/space/chatrl/users/wlw_temp/verl/verl/experimental/agent_loop/tool_call_test_cases.jsonl
   #--prompt-data /gfs/space/chatrl/users/wlw_temp/data/dapo_17k/data/dapo-math-17k-with-answer-label.jsonl
   #--prompt-data /gfs/space/chatrl/users/wlw_temp/wlw/data/slime/DeepCoder-Preview-Dataset_wlw/taco/train.jsonl
   --prompt-data /gfs/space/chatrl/users/hxh/data/math_data/dapo-math/prompts/dapo-math-17k_dedup_no_prompt.jsonl
   --input-key messages
   --label-key answer

   #I think we should return raw prompt in agentic rollout, 
   #all prompt initialization and formatting should be handled in agentic rollout?
   #--apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":true}'
   #--rollout-shuffle
   --balance-data
   --rm-type dapo
   --num-rollout 3000

   --rollout-batch-size ${rollout_batch_size}
   --n-samples-per-prompt ${rollout_n}
   --num-steps-per-rollout ${num_steps_per_rollout}
   --rollout-max-response-len ${max_resp_len}
   --rollout-temperature 1

   --save-debug-rollout-data /gfs/space/chatrl/users/wlw_temp/slime_code/logs/withouttool_Rollout_${TIMESTAMP}/exp1_rollout_{rollout_id}.pt
)

EVAL_ARGS=(
   --eval-interval 10
   #--eval-prompt-data humaneval /gfs/space/chatrl/users/wlw_temp/wlw/data/slime/humaneval_codeinmd/humaneval_codeinmd.jsonl
   #--eval-prompt-data dapo /gfs/space/chatrl/users/hxh/data/math_data/dapo-math/prompts/dapo-math-17k_dedup_no_prompt.jsonl
   #--eval-prompt-data tool_call /gfs/space/chatrl/users/wlw_temp/verl/verl/experimental/agent_loop/tool_call_test_cases.jsonl
   --eval-prompt-data aime24_avg8 /gfs/space/chatrl/users/wlw_temp/data/data_aime25/data/train-00000-of-00001_avg8.jsonl
   --eval-input-key messages
   --eval-label-key answer
   --n-samples-per-eval-prompt 1
   --eval-max-response-len ${max_resp_len}
   --eval-max-context-len ${max_context_len}
   --eval-temperature 1
   --eval-top-p 1
   #--skip-eval-before-train
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   # --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   # --decoder-last-pipeline-num-layers 23

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${max_context_len}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8

   # --sglang-max-running-requests 512
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32

   --log-probs-chunk-size 1024
)

CUSTOM_ARGS=(
   --custom-rm-path slime.rollout.rm_hub.remote_reward_model.remote_reward_function
   --custom-config-path examples/code/code.yaml
)

REMOTE_RM_ARGS=(
   --rm-api-key EMPTY
   --rm-base-url http://10.102.236.235:6669/v1
   --rm-model-name Qwen3-30B-A3B
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
        "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
        "PYTHONPATH": "/root/Megatron-LM/",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "^openib",
        "OMPI_MCA_routed": "direct",
        "OMPI_MCA_routed_radix": "1024",
        "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
        "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
        "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}",
        "RAY_DEBUG": "0"
     }
   }' \
   -- python3 train.py \
   --use-tensorboard \
   --tensorboard-dir ${TENSORBOARD_DIR} \
   --actor-num-nodes ${nnodes} \
   --num-gpus-per-node ${num_gpus_per_node} \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${REMOTE_RM_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]} $@ 2>&1 | tee ${LOG_FILE}

