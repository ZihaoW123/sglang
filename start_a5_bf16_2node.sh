#!/usr/bin/env bash
# 244: bash start_a5_bf16_2node.sh 0
# 248: bash start_a5_bf16_2node.sh 1
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

cd "$(dirname "$0")" || exit 1
NODE_RANK=${1:?Usage: bash start_a5_bf16_2node.sh 0_or_1}
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]] || exit 1
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="127.0.0.1,localhost,141.61.54.244,141.61.54.248,172.27.13.244,172.27.13.248,${no_proxy:-}"
export NO_PROXY="$no_proxy"
export SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_NPU_PROFILING=0
unset ASCEND_LAUNCH_BLOCKING
export HCCL_BUFFSIZE=256
export DEEPEP_HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=300
export HCCL_SOCKET_IFNAME=data0.3001
export GLOO_SOCKET_IFNAME=data0.3001
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/memfabric_hybrid/set_env.sh ]]; then
    source /usr/local/memfabric_hybrid/set_env.sh
fi

MODEL_PATH=/mnt/share/w00936111/weights/GLM-5.3-Flash-BF16
MEM_FRACTION=0.80
CUDA_GRAPH_BS="1 8 64"

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --dtype bfloat16 --kv-cache-dtype bf16 \
    --attention-backend ascend --device npu \
    --tp-size 16 --dp-size 1 --nnodes 2 --node-rank "$NODE_RANK" \
    --dist-init-addr 172.27.13.244:29500 \
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION" --page-size 64 \
    --served-model-name GLM-NEXT --load-format auto \
    --max-running-requests 16 \
    --speculative-draft-model-path "$MODEL_PATH" \
    --speculative-draft-kv-cache-dtype bf16 \
    --speculative-algorithm NEXTN --speculative-num-steps 4 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
    --pre-warm-nccl --watchdog-timeout 1200 \
    --cuda-graph-bs ${CUDA_GRAPH_BS} \
    --host 0.0.0.0 --port 8810
