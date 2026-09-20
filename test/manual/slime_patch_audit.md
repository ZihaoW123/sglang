# GLM-5.3-Flash / Slime compatibility port

This port starts from `glmx/glmx-main-a3-a5-merge` at `4f134951ef` and adapts
the Slime SGLang patch series in `slime-ascend/patches/third_party/sglang/`.
The two most recent commits on `ZihaoW123/sglang:ifmn/npu/glm-5-optim_0824`
(`24855140e4`, `bf1b3da3c8`) were used as the semantic reference. The patch
was integrated as source changes because SGLang has since reorganized the
runtime, especially the host KV cache classes and speculative decoding.

The port covers memory-saver graph arguments, idempotent release/resume,
optional HiCache host-memory release, `/post_process_weights`, `/pull_weights`
for full/delta checkpoints, top-p replay metadata and logprobs, PD timeouts and
diagnostics, DSA/indexer fixes, the draft CPU-offloader guard, and the
multimodal/NPU compatibility changes from the Slime patch set. The four-layer
GLM-5.3-Flash checkpoint uses symlinks to original safetensors shards; its
weight loader ignores later-layer tensors that share those shards.

Local checks:

```bash
python3 -m compileall -q python/sglang/srt
python3 -m ruff check --select F821,F822,F823 python/sglang/srt
```

The focused CPU test is `python test/manual/test_slime_memory_compat.py -v`.
It requires the SGLang runtime dependencies. NPU inference and Slime rollout
still require validation on the server; passing source-level checks alone does
not establish that the full RL flow works.
