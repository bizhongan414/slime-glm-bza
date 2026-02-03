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
export num_gpus_per_node=1
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

export train_file=/gfs/platform/public/infra/all_train_w_difficulty_testcase_max30.jsonl

export max_resp_len=$(( 1024 * 10 ))
export max_context_len=$(( 1024 * 10 ))
export rollout_batch_size=8
export global_batch_size=1
export rollout_num=1
export num_steps_per_rollout=1
export NCCL_GRAPH_REGISTER=0

# export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=True
# export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=1024

export EXP_NAME=for_sandbox_debug
export project_name=slime_30B-A3B_code_debug_single
export EXP_DIR=/gfs/space/chatrl/users/wlw_temp/slime_code
export DUMP_DIR=${EXP_DIR}/dump_details
export LOG_FILE=${EXP_DIR}/logs/output_${TIMESTAMP}.log
export CKPT_SAVE_PATH=${EXP_DIR}/checkpoints
export TENSORBOARD_DIR=${EXP_DIR}/tensorboard_log/GLM-4.7-Flash

export MODEL_PATH=/gfs/space/chatrl/public/models/Qwen3-4B
export DIST_MODEL_PATH=/gfs/space/chatrl/public/models/Qwen3-4Btorch_dist

bash /gfs/space/chatrl/users/wlw_temp/slime_code/slime/examples/code/test_scripts/run.sh