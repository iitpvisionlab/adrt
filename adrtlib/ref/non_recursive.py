from __future__ import annotations
from typing import Callable, TypeAlias
from adrtlib.ref.common import OpCount

Mid: TypeAlias = Callable[[int], int]  # calculate middle point using size


class Task:
    def __init__(
        self,
        start: int,
        stop: int,
        mid: Mid,
    ):
        """
        Single `Task` holds information about how to call a line like this:
        mergeHT(fht2dt(img[:n0], sign), fht2dt(img[n0:], sign))
        """
        assert stop > start, (stop, start)
        self.start = start
        self.stop = stop
        self.left_visited = False
        self.size = stop - start  # can be runtime evaluated
        self.mid = start + mid(self.size)  # can be runtime evaluated

    def __repr__(self):
        return (
            f"<{type(self).__name__} start={self.start} stop={self.stop} "
            f"left={self.left_visited}>"
        )

    def left(self, mid: Mid):
        return Task(self.start, self.mid, mid)

    def right(self, mid: Mid):
        return Task(self.mid, self.stop, mid)


def non_recursive(
    size: int, apply: Callable[[Task], OpCount], mid: Mid
) -> OpCount:
    """
    Technically, this implementation is worse than recursive, but it can be
    straightforwardly ported to compilable languages and be very efficient
    with 0 function calls.
    """
    tasks = [Task(0, size, mid)]
    task = tasks[-1]
    total_op_count = 0
    while True:
        while not task.left_visited:
            if task.size > 2:
                task = task.left(mid)
                tasks.append(task)
            else:
                task.left_visited = True
        if task.size > 1:
            total_op_count += apply(task)
        tasks.pop(-1)
        if not tasks:
            break
        task = tasks[-1]  # parent task
        if not task.left_visited:
            task.left_visited = True
            if task.size > 2:
                task = task.right(mid)
                tasks.append(task)
    return OpCount(total_op_count)
