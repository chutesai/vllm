# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental no-allocation decode interval cache for the qualified upstream revision.

Only prefix-cache-disabled, non-speculative single-token running requests
qualify. A slow allocation establishes capacity and the next eviction boundary.
All boundary steps and unsupported managers retain the upstream implementation.
"""

import weakref
from typing import TYPE_CHECKING, cast

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus

_ACTIVE_MANAGER: weakref.ReferenceType["DecodeSlotCache"] | None = None


if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class DecodeSlotCache(KVCacheManager):
    _decode_intervals: dict[str, tuple[Request, int, int, int]]
    _decode_shadow: bool
    _decode_hits: int
    _decode_misses: int
    _decode_shadow_hits: int

    def _request_snapshot(self, rid):
        return (
            self.block_pool.get_num_free_blocks(),
            tuple(
                (
                    tuple(
                        (b.block_id, b.ref_cnt, b.is_null) for b in g.req_to_blocks[rid]
                    ),
                    g.num_cached_block.get(rid),
                    getattr(g, "_num_checkpoint_blocks", {}).get(rid, 0),
                    getattr(g, "last_state_block_idx", {}).get(rid),
                    rid in getattr(g, "_allocated_block_reqs", ()),
                )
                for g in self.coordinator.single_type_managers
            ),
        )

    def _remember_interval(self, request):
        rid = request.request_id
        groups = self.coordinator.single_type_managers
        computed = request.num_computed_tokens
        processed = max(0, computed - request.num_in_flight_tokens)
        capacity = self.max_model_len
        for group in groups:
            if type(group).__name__ not in (
                "FullAttentionManager",
                "SlidingWindowManager",
                "MambaManager",
            ):
                return
            blocks = group.req_to_blocks.get(rid)
            if not blocks or blocks[-1].is_null or rid in group._partial_hit_reqs:
                return
            if getattr(group, "num_speculative_blocks", 0):
                return
            if getattr(group, "_num_checkpoint_blocks", {}).get(rid, 0):
                return
            capacity = min(capacity, len(blocks) * group.block_size)
        eviction_limit = capacity + 1
        for group in groups:
            block_size = group.block_size
            bucket = group.get_num_skipped_tokens(processed) // block_size
            # Supported eviction functions are monotone. Cache only while no
            # manager enters a new whole-block eviction bucket.
            lo, hi = processed + 1, capacity + 1
            if group.get_num_skipped_tokens(hi) // block_size != bucket:
                while lo < hi:
                    middle = (lo + hi) // 2
                    if group.get_num_skipped_tokens(middle) // block_size == bucket:
                        lo = middle + 1
                    else:
                        hi = middle
                eviction_limit = min(eviction_limit, lo)
        self._decode_intervals[rid] = (request, computed, capacity, eviction_limit)

    def allocate_slots(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
        skip_zeroing_group_ids=(),
    ):
        rid = request.request_id
        eligible = (
            not self.enable_caching
            and not skip_zeroing_group_ids
            and num_new_tokens == 1
            and not num_new_computed_tokens
            and new_computed_blocks is None
            and not num_lookahead_tokens
            and not num_external_computed_tokens
            and not delay_cache_blocks
            and not num_encoder_tokens
            and not full_sequence_must_fit
            and not reserved_blocks
            and request.status == RequestStatus.RUNNING
            and request.num_computed_tokens >= request.num_prompt_tokens
            and not self.coordinator.num_reprefillable_tokens
        )
        entry = self._decode_intervals.get(rid) if eligible else None
        hit = (
            entry is not None
            and entry[0] is request
            and request.num_computed_tokens >= entry[1]
            and request.num_computed_tokens + 1 <= entry[2]
            and max(0, request.num_computed_tokens - request.num_in_flight_tokens)
            < entry[3]
        )
        if hit and not self._decode_shadow:
            self._decode_hits += 1
            return self.empty_kv_cache_blocks
        snapshot = self._request_snapshot(rid) if hit else None
        # On a miss, invalidate before touching any cache-manager state.
        self._decode_intervals.pop(rid, None)
        out = super().allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
            skip_zeroing_group_ids=skip_zeroing_group_ids,
        )
        self._decode_misses += 1
        if hit:
            if out is None or any(out.blocks):
                raise AssertionError("Predicted no-allocation step allocated blocks")
            if self._request_snapshot(rid) != snapshot:
                raise AssertionError("Predicted no-op step changed cache bookkeeping")
            self._decode_shadow_hits += 1
        if out is not None and eligible:
            self._remember_interval(request)
        return out

    def free(self, request):
        self._decode_intervals.pop(request.request_id, None)
        return super().free(request)

    def pop_blocks_for_free(self, request):
        self._decode_intervals.pop(request.request_id, None)
        return super().pop_blocks_for_free(request)


class DecodeSlotScheduler(AsyncScheduler):
    shadow = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _ACTIVE_MANAGER
        manager = cast(DecodeSlotCache, self.kv_cache_manager)
        if manager.enable_caching:
            raise ValueError("Prefix caching must be disabled")
        manager.__class__ = DecodeSlotCache
        manager._decode_intervals = {}
        manager._decode_shadow = self.shadow
        manager._decode_hits = manager._decode_misses = manager._decode_shadow_hits = 0
        _ACTIVE_MANAGER = weakref.ref(manager)


class ShadowDecodeSlotScheduler(DecodeSlotScheduler):
    shadow = True


class DecodeSlotCacheWorker:
    model_runner: "GPUModelRunner"

    def parallax_set_decode_slot_cache(self, enabled, shadow=False):
        if self.model_runner.get_model().config.model_type != "parallax":
            raise ValueError("Experimental Parallax worker only")
        manager = _ACTIVE_MANAGER() if _ACTIVE_MANAGER is not None else None
        if manager is None or not enabled or manager._decode_shadow != shadow:
            raise ValueError("Custom decode-slot scheduler was not installed")
        self._decode_slot_manager = manager
        return {
            "enabled": enabled,
            "shadow": shadow,
            "groups": [
                (type(g).__name__, g.block_size, getattr(g, "mamba_cache_mode", None))
                for g in manager.coordinator.single_type_managers
            ],
        }

    def parallax_decode_slot_cache_report(self):
        manager = self._decode_slot_manager
        return {
            "hits": manager._decode_hits,
            "misses": manager._decode_misses,
            "shadow_hits": manager._decode_shadow_hits,
            "live_entries": len(manager._decode_intervals),
        }
