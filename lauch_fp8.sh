#!/usr/bin/env bash
cd /home/p00931643/glmx || exit 1
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="127.0.0.1,localhost,141.61.54.248,${no_proxy:-}"
export NO_PROXY="$no_proxy"
export SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_NPU_PROFILING=0
unset ASCEND_LAUNCH_BLOCKING
export SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1
export HCCL_BUFFSIZE=256
export DEEPEP_HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=300
export HCCL_SOCKET_IFNAME=data0.3001
export GLOO_SOCKET_IFNAME=data0.3001
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/memfabric_hybrid/set_env.sh ]]; then
  source /usr/local/memfabric_hybrid/set_env.sh
fi
MODEL_PATH=/mnt/share/w00936111/weights/GLM-5.3-Flash
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --dtype bfloat16 --quantization fp8 --kv-cache-dtype bf16 \
  --attention-backend ascend --device npu --tp-size 8 --nnodes 1 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --trust-remote-code --mem-fraction-static 0.84 --page-size 64 \
  --served-model-name GLM-NEXT --load-format auto --max-running-requests 16 \
  --moe-a2a-backend "${1:-deepep}" --deepep-mode auto \
  --speculative-draft-model-path "$MODEL_PATH" --speculative-draft-kv-cache-dtype bf16 \
  --speculative-draft-model-quantization fp8 --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --pre-warm-nccl --watchdog-timeout 1200 --random-seed 42 \
  --cuda-graph-bs 1 8 64 --host 0.0.0.0 --port 8810 "${@:2}"