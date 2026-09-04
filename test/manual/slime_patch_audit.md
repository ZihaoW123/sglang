# Slime 0.5.15.post1 compatibility audit for the GLM-5.2 NPU branch

## Scope and provenance

Audited against SGLang `24855140e4a50e71847f5c89ffc9b4324a746c25` and
slime-ascend `4a66e6b23edabfea10532bcb3a41bc14bbb78435` on 2026-09-04.
The six source patches are the `0001` through `0006` files under
`slime-ascend/patches/third_party/sglang/`; the separate
`sglang_tms.patch` behavior is included as well. The current SGLang branch has
refactored several of the affected modules, so this is a semantic port rather
than a mechanical patch application.

## Compatibility map

| Old patch area | Current implementation |
| --- | --- |
| Slime memory-saver CUDA-graph arguments | Positional and keyword arguments are forwarded by both real and no-op adapters. |
| Memory release/resume retries | Repeated and overlapping tag sets are idempotent; weight-cache safety checks happen before state changes. |
| HiCache host-memory release | MHA, MLA, K-only, DSA, Mamba and grouped host pools release and reconstruct buffers; radix caches delegate the lifecycle when L3 storage is disabled. Controlled by `--release-hicache`. |
| Draft-worker CPU offloader | Draft initialization does not replace the process-global target offloader. |
| Quantized-weight post-processing | `/post_process_weights` reaches current `WeightUpdater` and invokes restore/repack hooks inside the device-loading context. |
| Local full/delta checkpoints | `/pull_weights` supports host-locked full reset and ordered zstd XOR/overwrite deltas with checksum verification and an optional pre-read hook. |
| Top-p rollout replay | The legacy `custom_params.return_top_p_token_ids` request is propagated through normal, greedy, Ascend, speculative, pipeline-parallel and PD paths. Sparse supports are returned as base64 int32 values plus offsets. Selected-token logprobs are renormalized over the replay support, force-including a backend-selected boundary token. |
| PD failure handling | Bootstrap and transfer queues have bounded timeouts, abort/cleanup paths and failure metrics; current Mooncake already propagates extra-state transfer failures. |
| PD diagnostics | `/get_load` supports an `inflight` section and request time stats retain queue times, transfer statistics and retry counts. |
| DSA/indexer compatibility | Shared head gates expand to query-head count, `INDEXER_ROPE_NEOX_STYLE` is honored, shared-indexer query heads are repeated, index-key allocation is tagged as KV cache, and graph capture uses token rather than batch count. |
| EAGLE graph inputs | Draft top-k probabilities and ids are clamped before graph-buffer copies; speculative top-p metadata only records accepted positions. |
| NPU expert parallelism | The newer branch already creates a standalone NPU MoE EP group and the fused MoE dispatcher uses its device group. |
| Multimodal/EPD | GLM-4V precomputed embeddings and encoder/language-only loading, Qwen legacy loading, and Qwen3-VL deepstack guards are retained on the current interfaces. |
| Profiling, compressed tensors and FastAPI routing | Old endpoint, quantization and mounted-route compatibility fixes are carried over. |
| Cohere strict decorator | Already absent in the newer branch; no source change is needed. |

## Focused regression test

Run with the target SGLang installation:

```bash
python test/manual/test_slime_memory_compat.py -v
```

The CPU-only test covers graph-argument forwarding, idempotent lifecycle and
HiCache delegation, target/draft offloader isolation, weight-cache rejection,
top-p support reconstruction and wire encoding, quantization hooks, and a real
tiny zstd XOR checkpoint delta with checksum verification. Whole-package
`compileall`, undefined-symbol linting and the reduced GLM-5.2 NPU inference/RL
smoke tests complement it.

The reduced one-dense/three-sparse GLM-5.2 model also completed a TP8 Ascend
inference smoke test. A separate temperature-1, top-p-0.8, top-k-50 request
returned two completion tokens, two 50-token replay supports encoded as offsets
`[0, 50, 100]`, and finite selected-token logprobs. This caught and fixed a
missing ordinary-decode accumulation step that a tensor-level test alone did
not expose.

## Controlled RL reference

The earlier deterministic reduced-model comparison used one dense plus three
shared-indexer sparse layers, EAGLE, DP/EP 8, colocation, BF16 and TIS. Before
and after the initial memory-lifecycle compatibility port, all eight generated
sequences and saved token/logprob tensors were exactly equal. Both runs reported:

| Metric | Value |
| --- | ---: |
| `train_rollout_logprob_abs_diff` | 0.3517397344112396 |
| `tis_abs` | 0.30914774537086487 |
| `tis` | 0.9672589302062988 |
| `tis_clipfrac` | 0.134765625 |
| `entropy_loss` | 9.891523361206055 |

That reference used top-p 1.0 and disabled graph capture, so it demonstrates no
regression for the original lifecycle subset, not an accuracy claim for every
newly ported path. Top-p below 1, PD, multimodal, HiCache, local delta loading,
and graph/R3 paths are feature-specific compatibility surfaces and should be
validated in their actual deployment modes.
