#!/usr/bin/env bash
# Run inside 248 / pxc_glm53_flash from /home/p00931643/glmx.
cd "$(dirname "$0")" || exit 1
MODEL_PATH=/mnt/share/w00936111/weights/GLM-5.3-Flash-0day-A5-convert
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="127.0.0.1,localhost,141.61.54.248,141.61.54.244,${no_proxy:-}"
export NO_PROXY="$no_proxy"
python3 -m sglang.benchmark.serving \
    --backend sglang \
    --host 141.61.54.248 --port 8000 \
    --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 1 --gsp-prompts-per-group 64 \
    --gsp-system-prompt-len 117964 --gsp-question-len 13107 \
    --gsp-output-len 1024 \
    --num-prompts 64 --max-concurrency 8 --request-rate inf \
    --cache-report \
    --output-file "${1:-pd_128k_1k_cache90.jsonl}"
