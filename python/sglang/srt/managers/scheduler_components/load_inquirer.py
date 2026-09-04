from __future__ import annotations

import logging
import time
from itertools import islice
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.load_snapshot import (
    DisaggregationMetrics,
    LoadSnapshot,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.runtime_context import get_lora

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from sglang.srt.managers.tp_worker import BaseTpWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


logger = logging.getLogger(__name__)

# Load snapshots use a fixed 16 KiB shared-memory slot. Keep diagnostics useful
# without allowing a large request queue to break periodic load publication.
MAX_INFLIGHT_REQS_PER_SNAPSHOT = 32
MAX_INFLIGHT_ID_CHARS = 128


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerLoadInquirer:
    disaggregation_mode: DisaggregationMode
    ps: ParallelState
    server_args: ServerArgs
    max_total_num_tokens: int
    max_running_requests: int
    pool_stats_observer: SchedulerPoolStatsObserver
    tp_worker: BaseTpWorker
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    spec_algorithm: SpeculativeAlgorithm
    get_running_batch: Callable
    get_waiting_queue: Callable
    get_stats: Callable
    get_chunked_req: Callable
    get_disagg_prefill_bootstrap_queue: Callable
    get_disagg_prefill_inflight_queue: Callable
    get_disagg_decode_prealloc_queue: Callable
    get_disagg_decode_transfer_queue: Callable
    get_spec_total_num_accept_tokens: Callable
    get_spec_total_num_forward_ct: Callable
    get_total_prefill_uncached_tokens: Callable
    get_total_prefill_busy_us: Callable
    get_decode_moment_totals: Callable

    def _get_num_pending_tokens(self, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_indices`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_indices`` is already up-to-date, so the default
                0 is correct.
        """
        num_pending_tokens = sum(req.seqlen for req in self.get_waiting_queue())
        if self.get_chunked_req() is not None:
            req = self.get_chunked_req()
            num_pending_tokens += req.seqlen - len(req.prefix_indices) - chunk_deduct
        return num_pending_tokens

    def get_num_waiting_uncached_tokens(self) -> int:
        """Get uncached input tokens waiting for prefill compute."""
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return 0
        num_tokens = 0
        for req in self.get_waiting_queue():
            # if match-in-waiting-queue disabled, this metric returns seq_lens
            num_tokens += max(0, req.seqlen - req.num_matched_prefix_tokens)
        cr = self.get_chunked_req()
        if cr is not None:
            num_tokens += max(0, cr.seqlen - len(cr.prefix_indices))
        return num_tokens

    def get_loads(self) -> LoadSnapshot:
        """Build the per-DP-rank load snapshot for DP balancing and /v1/loads."""
        stats = self.get_stats()
        num_running_reqs = len(self.get_running_batch().reqs)

        waiting_queues = [self.get_waiting_queue()]
        pending_token_queues = [self.get_waiting_queue()]
        awaiting_kv_tokens = 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            prefill_bootstrap_queue = self.get_disagg_prefill_bootstrap_queue().queue
            waiting_queues.append(prefill_bootstrap_queue)
            pending_token_queues.append(prefill_bootstrap_queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            decode_prealloc_queue = self.get_disagg_decode_prealloc_queue().queue
            decode_transfer_queue = self.get_disagg_decode_transfer_queue().queue
            decode_retracted_queue = (
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )
            waiting_queues.append(decode_prealloc_queue)
            waiting_queues.append(decode_transfer_queue)
            waiting_queues.append(decode_retracted_queue)
            # In disaggregated decode, transfer-queue requests and transferred
            # waiting-queue requests have already pre-allocated decode-side KV
            # slots, so they are already included in num_used_tokens.
            pending_token_queues = [decode_prealloc_queue, decode_retracted_queue]
            # KV not yet arrived from the prefill side.
            awaiting_kv_tokens = sum(
                req.seqlen
                for queue in (decode_prealloc_queue, decode_transfer_queue)
                for req in queue
            )

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_used_tokens, kv_token_usage = (
            self.pool_stats_observer.get_pool_stats().get_kv_token_stats()
        )
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in pending_token_queues for req in queue
        )
        num_active_tokens = max(0, num_total_tokens - awaiting_kv_tokens)

        memory = None
        try:
            memory = MemoryMetrics(
                weight_gb=round(self.tp_worker.model_runner.weight_load_mem_usage, 3),
                kv_cache_gb=round(
                    self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                ),
                graph_gb=round(sum(self.tp_worker.graph_memory_usage.values()), 3),
                token_capacity=int(self.max_total_num_tokens),
            )
        except (AttributeError, TypeError) as e:
            logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if (
            not self.spec_algorithm.is_none()
            and self.get_spec_total_num_forward_ct() > 0
        ):
            speculative = SpeculativeMetrics(
                accept_length=(
                    self.get_spec_total_num_accept_tokens()
                    / self.get_spec_total_num_forward_ct()
                ),
                accept_rate=stats.spec_accept_rate,
            )

        lora = None
        if get_lora().enable_lora:
            lora = LoRAMetrics(
                slots_used=stats.lora_pool_slots_used,
                slots_total=stats.lora_pool_slots_total,
                utilization=stats.lora_pool_utilization,
            )

        mode_str = "null"
        prefill_bootstrap = prefill_inflight = 0
        decode_prealloc = decode_transfer = decode_retracted = 0
        decode_prealloc_ready = 0
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            mode_str = "prefill"
            prefill_bootstrap = len(self.get_disagg_prefill_bootstrap_queue().queue)
            prefill_inflight = len(self.get_disagg_prefill_inflight_queue())
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            mode_str = "decode"
            decode_prealloc = len(self.get_disagg_decode_prealloc_queue().queue)
            decode_transfer = len(self.get_disagg_decode_transfer_queue().queue)
            decode_retracted = len(
                self.get_disagg_decode_prealloc_queue().retracted_queue
            )
            decode_prealloc_ready = sum(
                1
                for decode_req in self.get_disagg_decode_prealloc_queue().queue
                if decode_req.waiting_for_input
            )
        disaggregation = DisaggregationMetrics(
            mode=mode_str,
            prefill_bootstrap_queue_reqs=prefill_bootstrap,
            prefill_inflight_queue_reqs=prefill_inflight,
            decode_prealloc_queue_reqs=decode_prealloc,
            decode_transfer_queue_reqs=decode_transfer,
            decode_retracted_queue_reqs=decode_retracted,
            kv_transfer_speed_gb_s=stats.kv_transfer_speed_gb_s,
            kv_transfer_latency_ms=stats.kv_transfer_latency_ms,
        )

        queues = QueueMetrics(
            waiting=len(self.get_waiting_queue()),
            grammar=stats.num_grammar_queue_reqs,
            paused=stats.num_paused_reqs,
            retracted=stats.num_retracted_reqs,
            prealloc_ready=decode_prealloc_ready,
        )

        totals = self.get_decode_moment_totals()
        decode_moments = list(totals) if totals[0] > 0 else None

        now_perf = time.perf_counter()
        inflight_queues = [("running", self.get_running_batch().reqs, None)]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            inflight_queues.extend(
                [
                    ("waiting", self.get_waiting_queue(), "wait_queue_entry_time"),
                    (
                        "bootstrap",
                        self.get_disagg_prefill_bootstrap_queue().queue,
                        "prefill_bootstrap_queue_entry_time",
                    ),
                    (
                        "prefill_inflight",
                        self.get_disagg_prefill_inflight_queue(),
                        "prefill_transfer_queue_entry_time",
                    ),
                ]
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            prealloc = self.get_disagg_decode_prealloc_queue()
            inflight_queues.extend(
                [
                    ("waiting", self.get_waiting_queue(), "wait_queue_entry_time"),
                    ("prealloc", prealloc.queue, "decode_prealloc_queue_entry_time"),
                    (
                        "transfer",
                        self.get_disagg_decode_transfer_queue().queue,
                        "decode_transfer_queue_entry_time",
                    ),
                    (
                        "retracted",
                        prealloc.retracted_queue,
                        "decode_prealloc_queue_entry_time",
                    ),
                ]
            )
        else:
            inflight_queues.append(
                ("waiting", self.get_waiting_queue(), "wait_queue_entry_time")
            )

        def bounded_id(value):
            if value is None:
                return None
            return str(value)[:MAX_INFLIGHT_ID_CHARS]

        def describe(entry, stage, entry_time_field):
            req = getattr(entry, "req", entry)
            item = {
                "rid": bounded_id(getattr(req, "rid", None)),
                "bootstrap_room": bounded_id(getattr(req, "bootstrap_room", None)),
                "seqlen": getattr(entry, "seqlen", getattr(req, "seqlen", None)),
                "stage": stage,
            }
            if entry_time_field is not None:
                stats = getattr(req, "time_stats", None)
                entry_time = getattr(stats, entry_time_field, 0.0) if stats else 0.0
                item["age_s"] = (
                    round(now_perf - entry_time, 3) if entry_time else None
                )
            if entry is not req:
                item["waiting_for_input"] = getattr(
                    entry, "waiting_for_input", None
                )
                item["timeout_cancel_issued"] = getattr(
                    entry, "timeout_cancel_issued", None
                )
            return item

        inflight = []
        remaining = MAX_INFLIGHT_REQS_PER_SNAPSHOT
        for name, queue, field in inflight_queues:
            num_reqs = len(queue)
            sample = list(islice(iter(queue), remaining))
            remaining -= len(sample)
            inflight.append(
                {
                    "name": name,
                    "num_reqs": num_reqs,
                    "reqs": [describe(entry, name, field) for entry in sample],
                    "truncated": max(0, num_reqs - len(sample)),
                }
            )

        return LoadSnapshot(
            dp_rank=int(self.ps.dp_rank) if self.ps.dp_rank is not None else 0,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_waiting_uncached_tokens=self.get_num_waiting_uncached_tokens(),
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            num_active_tokens=num_active_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(stats.gen_throughput, 2),
            cache_hit_rate=round(stats.cache_hit_rate, 4),
            utilization=round(stats.utilization, 4),
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
            total_prefill_uncached_tokens=self.get_total_prefill_uncached_tokens(),
            total_prefill_busy_us=self.get_total_prefill_busy_us(),
            decode_moments=decode_moments,
            inflight=inflight,
        )
