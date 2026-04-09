"""US4 tests for RingBuffer behavior and spawn safety."""

import time

from src.utils.multiprocessing import RingBuffer, get_spawn_context


def test_ring_buffer_wraparound_latest_value() -> None:
    buffer = RingBuffer(size=3)

    buffer.put({"step": 1})
    buffer.put({"step": 2})
    buffer.put({"step": 3})
    buffer.put({"step": 4})

    latest = buffer.get_latest()
    assert latest is not None
    assert latest["step"] == 4


def _spawn_writer(shared_buffer: RingBuffer) -> None:
    shared_buffer.put({"source": "spawn-worker", "value": 99})


def test_ring_buffer_spawn_safe_put_get_integration() -> None:
    buffer = RingBuffer(size=15)
    ctx = get_spawn_context()

    proc = ctx.Process(target=_spawn_writer, args=(buffer,))
    proc.start()

    deadline = time.time() + 10.0
    while proc.exitcode is None and time.time() < deadline:
        proc.join(timeout=0.1)

    if proc.exitcode is None:
        proc.terminate()
        proc.join(timeout=1)
    assert proc.exitcode == 0

    latest = buffer.get_latest()
    assert latest is not None
    assert latest["source"] == "spawn-worker"
    assert latest["value"] == 99
