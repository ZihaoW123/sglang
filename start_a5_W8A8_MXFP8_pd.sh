#!/usr/bin/env bash
# Run in glmx: bash start_a5_W8A8_MXFP8_pd.sh
# P: 248 / pxc_glm53_flash; D: 244 / pxc_glm53_flash_v2
# Router on 248: bash start_a5_W8A8_MXFP8_pd.sh router

cd "$(dirname "$0")" || exit 1
MODEL_PATH=/mnt/share/w00936111/weights/GLM-5.3-Flash-0day-A5-convert
PREFILL_IP=141.61.54.248
DECODE_IP=141.61.54.244
LOCAL_IPS=" $(hostname -I) "
export no_proxy="127.0.0.1,localhost,$PREFILL_IP,$DECODE_IP,${no_proxy:-}"
export NO_PROXY="$no_proxy"

if [[ "${1:-}" == router ]]; then
    exec python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "http://$PREFILL_IP:8810" 8998 \
        --decode "http://$DECODE_IP:8811" \
        --host "$PREFILL_IP" --port 8000 \
        --policy round_robin \
        --worker-startup-timeout-secs 3600 \
        --request-timeout-secs 7200 \
        --disable-retries
fi

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_NPU_PROFILING=0
export HCCL_BUFFSIZE=256
export ASCEND_MF_STORE_URL="tcp://$PREFILL_IP:24670"
export ASCEND_MF_TRANSFER_PROTOCOL=device_urma
unset ASCEND_LAUNCH_BLOCKING

# TP8 / DP2 => attention TP4 per group.
# SGLang divides chunked-prefill-size by DP2: 16384 => 8192 per group.
if [[ "$LOCAL_IPS" == *" $PREFILL_IP "* ]]; then
    export SGLANG_HOST_IP="$PREFILL_IP"
    exec python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --quantization modelslim --load-format auto \
        --device npu --attention-backend ascend --trust-remote-code \
        --served-model-name GLM-NEXT \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port 8998 \
        --disaggregation-transfer-backend ascend \
        --tp-size 8 --pp-size 1 --dp-size 2 --enable-dp-attention --nnodes 1 \
        --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
        --mem-fraction-static 0.83 --page-size 64 --max-running-requests 16 \
        --speculative-draft-model-path "$MODEL_PATH" \
        --speculative-draft-kv-cache-dtype bf16 \
        --speculative-algorithm NEXTN --speculative-num-steps 4 \
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
        --pre-warm-nccl --watchdog-timeout 1200 --cuda-graph-bs 1 8 64 \
        --host "$PREFILL_IP" --port 8810
elif [[ "$LOCAL_IPS" == *" $DECODE_IP "* ]]; then
    export SGLANG_HOST_IP="$DECODE_IP"
    exec python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --quantization modelslim --load-format auto \
        --device npu --attention-backend ascend --trust-remote-code \
        --served-model-name GLM-NEXT \
        --disaggregation-mode decode \
        --disaggregation-bootstrap-port 8998 \
        --disaggregation-transfer-backend ascend \
        --tp-size 8 --pp-size 1 --dp-size 2 --enable-dp-attention --nnodes 1 \
        --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
        --mem-fraction-static 0.83 --page-size 64 --max-running-requests 16 \
        --speculative-draft-model-path "$MODEL_PATH" \
        --speculative-draft-kv-cache-dtype bf16 \
        --speculative-algorithm NEXTN --speculative-num-steps 4 \
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
        --pre-warm-nccl --watchdog-timeout 1200 --cuda-graph-bs 1 8 64 \
        --host "$DECODE_IP" --port 8811
else
    echo "Expected local IP $PREFILL_IP or $DECODE_IP; got:$LOCAL_IPS" >&2
    exit 1
fi
