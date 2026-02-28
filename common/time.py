import time


def now_ms() -> int:
    return time.perf_counter_ns() // 1_000_000
