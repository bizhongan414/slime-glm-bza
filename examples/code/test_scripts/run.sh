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
   --rollout-function-path examples.code.custom_multi_turn.generate_rollout
   --prompt-data /gfs/space/chatrl/users/wlw_temp/verl/verl/experimental/agent_loop/tool_call_test_cases.jsonl
   --input-key messages
   --label-key answer

   #I think we should return raw prompt in agentic rollout, 
   #all prompt initialization and formatting should be handled in agentic rollout?
   #--apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":"True"}'
   #--rollout-shuffle
   --balance-data
   --rm-type dapo
   --num-rollout 3000

   --rollout-batch-size ${rollout_batch_size}
   --n-samples-per-prompt ${rollout_n}
   --num-steps-per-rollout ${num_steps_per_rollout}
   --rollout-max-response-len ${max_resp_len}
   --rollout-temperature 1
)

EVAL_ARGS=(
#    --eval-interval 20
   --eval-prompt-data humaneval /gfs/space/chatrl/users/wlw_temp/wlw/data/slime/humaneval_codeinmd/humaneval_codeinmd.jsonl
   #--eval-prompt-data dapo /gfs/space/chatrl/users/hxh/data/math_data/dapo-math/prompts/dapo-math-17k_dedup_no_prompt.jsonl
   --eval-input-key prompt
   --n-samples-per-eval-prompt 1
   --eval-max-response-len ${max_resp_len}
   --eval-max-context-len ${max_context_len}
   --eval-temperature 1
   --eval-top-p 1
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
   --sglang-mem-fraction-static 0.7

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
   #--custom-rm-path examples.code.single_turn_reward_fn.reward_fn
   --custom-config-path examples/code/code.yaml
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
        "RAY_DEBUG": "1"
     }
   }' \
   -- python3 train.py \
   --use-tensorboard \
   --tensorboard-dir ${TENSORBOARD_DIR} \
   --actor-num-nodes ${nnodes} \
   --num-gpus-per-node ${num_gpus_per_node} \
   --actor-num-gpus-per-node 1 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]} $@ 2>&1 | tee ${LOG_FILE}

