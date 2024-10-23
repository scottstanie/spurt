import math
from collections.abc import Callable
from concurrent.futures import Executor, Future
from multiprocessing import Lock, cpu_count
from pathlib import Path
from typing import Optional, ParamSpec, TypeVar


def get_cpu_count() -> int:
    """Get the number of CPUs available to the current process.

    This function accounts for the possibility of a Docker container with
    limited CPU resources on a larger machine (which is ignored by
    `multiprocessing.cpu_count()`). This is derived from
    isce-framework/dolphin.

    Returns
    -------
    int
        The number of CPUs available to the current process.
    """

    def get_cpu_quota() -> int:
        return int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())

    def get_cpu_period() -> int:
        return int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())

    try:
        cfs_quota_us = get_cpu_quota()
        cfs_period_us = get_cpu_period()
        if cfs_quota_us > 0 and cfs_period_us > 0:
            return int(math.ceil(cfs_quota_us / cfs_period_us))
    except Exception:
        pass
    return cpu_count()


# Used for callable types
T = TypeVar("T")
P = ParamSpec("P")


class DummyProcessPoolExecutor(Executor):
    """Dummy ProcessPoolExecutor for to avoid forking for single_job purposes."""

    def __init__(self, max_workers: Optional[int] = None, **kwargs):  # noqa: ARG002
        self._max_workers = max_workers
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        with self._shutdownLock:
            if self._shutdown:
                msg = "Cannot schedule new futures after shutdown"
                raise RuntimeError(msg)

            future: Future = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                future.set_exception(e)
            else:
                future.set_result(result)

            return future

    def shutdown(
        self,
        wait: bool = True,  # noqa: FBT001, FBT002, ARG002
        cancel_futures: bool = True,  # noqa: FBT001, FBT002, ARG002
    ):
        with self._shutdownLock:
            self._shutdown = True

    def map(self, fn: Callable[P, T], *iterables, **kwargs):  # noqa: ARG002
        return map(fn, *iterables)
