"""Focused regression tests for legacy Slime compatibility (CPU-only)."""

import base64
import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
import zlib
from array import array
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler_components import weight_updater


class TestSlimeMemoryCompatibility(unittest.TestCase):
    def test_graph_arguments_reach_memory_saver(self):
        calls = []

        @contextmanager
        def cuda_graph(*args, **kwargs):
            calls.append((args, kwargs))
            yield

        fake = types.ModuleType("torch_memory_saver")
        fake.torch_memory_saver = types.SimpleNamespace(cuda_graph=cuda_graph)
        import sglang.srt.utils.torch_memory_saver_adapter as installed_adapter

        spec = importlib.util.spec_from_file_location(
            "_slime_adapter_test", Path(installed_adapter.__file__)
        )
        adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch_memory_saver": fake}):
            spec.loader.exec_module(adapter)
        graph, pool = object(), object()
        with adapter._TorchMemorySaverAdapterReal().cuda_graph(graph, pool=pool):
            pass
        self.assertEqual(calls, [((graph,), {"pool": pool})])
        with adapter._TorchMemorySaverAdapterNoop().cuda_graph(graph, pool=pool):
            pass
        self.assertEqual(len(calls), 1)

    def make_manager(self):
        return weight_updater.SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=Mock(),
            flush_cache=Mock(return_value=True),
            is_fully_idle=lambda: True,
        )

    def test_draft_preserves_target_offloader(self):
        from sglang.srt.model_executor import model_runner

        target = types.SimpleNamespace(
            is_draft_worker=False, ps=types.SimpleNamespace(dp_rank=3)
        )
        draft = types.SimpleNamespace(is_draft_worker=True)
        args, offloader = object(), object()
        with (
            patch.object(
                model_runner,
                "create_offloader_from_server_args",
                return_value=offloader,
            ) as create,
            patch.object(model_runner, "set_offloader") as install,
        ):
            model_runner.ModelRunner.init_cpu_offloader(target, args)
            model_runner.ModelRunner.init_cpu_offloader(draft, args)
        create.assert_called_once_with(args, dp_rank=3)
        install.assert_called_once_with(offloader)

    def test_weight_cache_rejection_does_not_change_offload_state(self):
        manager = self.make_manager()
        manager.tp_worker = Mock()
        manager.tp_worker.model_runner.server_args.weight_cache_mode = "consumer"
        tag = GPU_MEMORY_TYPE_WEIGHTS
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "weight cache is active"):
                manager.release_memory_occupation(
                    ReleaseMemoryOccupationReqInput(tags=[tag])
                )
            self.assertEqual(manager.offload_tags, set())
        manager.offload_tags.add(tag)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "weight cache is active"):
                manager.resume_memory_occupation(
                    ResumeMemoryOccupationReqInput(tags=[tag])
                )
            self.assertEqual(manager.offload_tags, {tag})
        manager.memory_saver_adapter.pause.assert_not_called()
        manager.memory_saver_adapter.resume.assert_not_called()

    @patch.object(weight_updater.torch, "get_device_module")
    def test_repeated_release_and_resume_are_idempotent(self, device_module):
        manager = self.make_manager()
        tag = GPU_MEMORY_TYPE_KV_CACHE
        release = ReleaseMemoryOccupationReqInput(tags=[tag, tag])
        resume = ResumeMemoryOccupationReqInput(tags=[tag, tag])
        manager.release_memory_occupation(release)
        manager.release_memory_occupation(release)
        manager.memory_saver_adapter.pause.assert_called_once_with(tag)
        manager.flush_cache.assert_called_once()
        manager.resume_memory_occupation(resume)
        manager.resume_memory_occupation(resume)
        manager.memory_saver_adapter.resume.assert_called_once_with(tag)
        self.assertEqual(manager.offload_tags, set())

    @patch.object(weight_updater.torch, "get_device_module")
    def test_overlapping_tags_only_transition_once(self, device_module):
        manager = self.make_manager()
        kv, graph = GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH
        manager.release_memory_occupation(ReleaseMemoryOccupationReqInput(tags=[kv]))
        manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=[kv, graph])
        )
        self.assertEqual(manager.memory_saver_adapter.pause.call_count, 2)
        self.assertEqual(manager.offload_tags, {kv, graph})
        manager.resume_memory_occupation(ResumeMemoryOccupationReqInput(tags=[kv]))
        manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=[kv, graph])
        )
        self.assertEqual(manager.memory_saver_adapter.resume.call_count, 2)
        self.assertEqual(manager.offload_tags, set())

    @patch.object(weight_updater.torch, "get_device_module")
    def test_hicache_lifecycle_is_delegated(self, device_module):
        manager = self.make_manager()
        tree_cache = Mock()
        manager.scheduler = types.SimpleNamespace(
            disaggregation_mode=None,
            server_args=types.SimpleNamespace(release_hicache=True),
            tree_cache=tree_cache,
        )
        req = ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        manager.release_memory_occupation(req)
        tree_cache.release_memory_occupation.assert_called_once_with()
        manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        )
        tree_cache.resume_memory_occupation.assert_called_once_with()


class TestSlimeTopPReplay(unittest.TestCase):
    def test_support_and_renormalized_logprob_include_sampled_token(self):
        from sglang.srt.layers.logprob_processor import (
            get_top_p_token_ids_from_probs,
            renorm_logprob_over_top_p,
        )

        probs = torch.tensor([[0.50, 0.30, 0.15, 0.05]], dtype=torch.float32)
        top_ks = torch.tensor([4], dtype=torch.int32)
        top_ps = torch.tensor([0.70], dtype=torch.float32)
        min_ps = torch.tensor([0.0], dtype=torch.float32)
        request_mask = torch.tensor([True])
        support = get_top_p_token_ids_from_probs(
            probs, top_ks, top_ps, min_ps, True, False, request_mask
        )
        self.assertEqual(support[0].tolist(), [0, 1])

        logprobs = renorm_logprob_over_top_p(
            probs,
            top_ks,
            top_ps,
            min_ps,
            True,
            False,
            request_mask,
            force_keep_token_ids=torch.tensor([3]),
        )
        self.assertTrue(torch.isfinite(logprobs[0, [0, 1, 3]]).all())
        self.assertTrue(torch.isneginf(logprobs[0, 2]))
        self.assertAlmostEqual(
            torch.exp(logprobs[0, [0, 1, 3]]).sum().item(), 1.0, places=6
        )

    def test_wire_encoding_is_flat_int32_with_offsets(self):
        from sglang.srt.managers.tokenizer_manager import _encode_top_p_token_ids

        token_blob, offset_blob = _encode_top_p_token_ids([[7, 9], [2]])
        token_ids = array("i")
        token_ids.frombytes(base64.b64decode(token_blob))
        offsets = array("i")
        offsets.frombytes(base64.b64decode(offset_blob))
        self.assertEqual(token_ids.tolist(), [7, 9, 2])
        self.assertEqual(offsets.tolist(), [0, 2, 3])

    def test_decode_result_accumulates_support_for_each_token(self):
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        logprob = types.SimpleNamespace(
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_p_token_ids=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
            top_logprobs_num=0,
            token_ids_logprob=None,
        )
        req = types.SimpleNamespace(logprob=logprob)
        batch = types.SimpleNamespace(
            spec_algorithm=types.SimpleNamespace(is_none=lambda: True)
        )
        output = types.SimpleNamespace(next_token_top_p_token_ids=[[1, 7, 9]])
        SchedulerBatchResultProcessor._apply_decode_logprobs(
            None,
            req=req,
            i=0,
            batch=batch,
            next_token_id=[7],
            next_token_logprobs=[-1.25],
            logits_output=output,
        )
        self.assertEqual(logprob.output_top_p_token_ids, [[1, 7, 9]])


class TestSlimeLoadDiagnostics(unittest.TestCase):
    def test_inflight_snapshot_round_trip_and_filtering(self):
        from sglang.srt.managers.load_snapshot import (
            SLOT_SIZE,
            LoadSnapshot,
            snapshot_decoder,
            snapshot_encoder,
        )

        inflight = [
            {
                "name": "running",
                "num_reqs": 1000,
                "reqs": [
                    {"rid": "r" * 128, "bootstrap_room": "b" * 128, "stage": "running"}
                    for _ in range(32)
                ],
                "truncated": 968,
            }
        ]
        snapshot = LoadSnapshot(dp_rank=3, inflight=inflight)
        payload = snapshot_encoder.encode(snapshot)
        self.assertLess(len(payload), SLOT_SIZE)
        decoded = snapshot_decoder.decode(payload)
        self.assertEqual(decoded.inflight, inflight)
        self.assertNotIn("inflight", decoded.to_dict({"core"}))
        self.assertEqual(decoded.to_dict({"inflight"})["inflight"], inflight)


class TestSlimeWeightHooks(unittest.TestCase):
    def test_quantization_post_process_hooks(self):
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            WeightUpdater,
        )

        calls = []

        class QuantMethod:
            def restore_weights_before_loading(self, module):
                calls.append(("restore", module))

            def process_weights_after_loading(self, module):
                calls.append(("process", module))

        module = torch.nn.Linear(2, 2)
        module.quant_method = QuantMethod()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(weight_cache_mode="off")
        )
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: module,
            update_model_fields=lambda **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: runner,
        )
        success, _ = updater.post_process_weights(True, True)
        self.assertTrue(success)
        self.assertEqual([name for name, _ in calls], ["restore", "process"])

    def test_local_checkpoint_xor_delta(self):
        from sglang.srt.weight_sync import local_checkpoint

        def write_safetensors(path, data, metadata=None):
            header = {
                "weight": {
                    "dtype": "U8",
                    "shape": [len(data)],
                    "data_offsets": [0, len(data)],
                }
            }
            if metadata is not None:
                header["__metadata__"] = metadata
            raw_header = json.dumps(header, separators=(",", ":")).encode()
            path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + data)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            local = root / "local"
            source = root / "source"
            delta_dir = source / "weight_v000001"
            base.mkdir()
            delta_dir.mkdir(parents=True)
            original = bytes([1, 2, 3, 4])
            updated = bytes([3, 6, 3, 9])
            write_safetensors(base / "model.safetensors", original)
            delta = bytes(a ^ b for a, b in zip(original, updated))
            compressed = local_checkpoint.zstandard.ZstdCompressor().compress(delta)
            checksum = f"{zlib.adler32(updated):08x}"
            write_safetensors(
                delta_dir / "delta.safetensors",
                compressed,
                metadata={"weight": checksum},
            )
            (delta_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": 1,
                            "base_version": 0,
                            "compression_format": "zstd",
                            "delta_encoding": "xor",
                            "checksum_format": "adler32",
                        }
                    }
                )
            )

            local_checkpoint.pull(str(local), str(base), str(source), 1)
            locations = local_checkpoint._tensor_locations(str(local))
            path, offset, size = locations["weight"]
            with open(path, "rb") as f:
                f.seek(offset)
                self.assertEqual(f.read(size), updated)
            self.assertEqual(local_checkpoint._read_applied_version(str(local)), 1)


if __name__ == "__main__":
    unittest.main()
