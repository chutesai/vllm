# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parallax integration contract regressions."""

from types import SimpleNamespace


def test_connector_zeroing_request_cannot_use_cached_noop(monkeypatch):
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.parallax_decode import DecodeSlotCache
    from vllm.v1.request import RequestStatus

    request = SimpleNamespace(
        request_id="r",
        status=RequestStatus.RUNNING,
        num_computed_tokens=10,
        num_prompt_tokens=1,
        num_in_flight_tokens=0,
    )
    manager = object.__new__(DecodeSlotCache)
    manager.enable_caching = False
    manager.coordinator = SimpleNamespace(num_reprefillable_tokens=0)
    manager._decode_intervals = {"r": (request, 1, 128, 128)}
    manager._decode_shadow = False
    manager._decode_hits = manager._decode_misses = 0
    called = []

    def allocate(self, *args, **kwargs):
        called.append(kwargs["skip_zeroing_group_ids"])
        return None

    monkeypatch.setattr(KVCacheManager, "allocate_slots", allocate)
    assert manager.allocate_slots(request, 1, skip_zeroing_group_ids=(2,)) is None
    assert called == [(2,)]
    assert manager._decode_hits == 0
    assert manager._decode_misses == 1
    assert "r" not in manager._decode_intervals
