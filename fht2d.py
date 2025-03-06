"""
https://doi.org/10.31857/S0132347421050022
"""

from common import ADRTResult, Image, Sign, add, round05


def fht2ds(img: Image, sign: Sign) -> ADRTResult:
    n = len(img)
    if n < 2:
        return ADRTResult(image=img, op_count=0)
    n0 = n // 2
    return mergeHT(fht2ds(img[:n0], sign), fht2ds(img[n0:], sign), sign)


def div_by_pow2(n: int) -> int:
    if n & (n - 1) == 0:
        return n // 2
    return 1 << (n.bit_length() - 1)


def fht2dt(img: Image, sign: Sign) -> ADRTResult:
    """
    Same as fht2ds, but division is done in powers of 2
    """
    n = len(img)
    if n < 2:
        return ADRTResult(img, op_count=0)
    n0 = div_by_pow2(n)
    return mergeHT(fht2dt(img[:n0], sign), fht2dt(img[n0:], sign), sign)


class Task:
    def __init__(
        self,
        start: int,
        stop: int,
        left_visited: bool = False,
        right_visited: bool = False,
    ):
        """
        Single `Task` holds information about how to call this line:
        mergeHT(fht2dt(img[:n0], sign), fht2dt(img[n0:], sign))
        """
        self.start = start
        self.stop = stop
        self.left_visited = left_visited
        self.right_visited = right_visited

    @property
    def size(self):
        assert self.stop > self.start, (self.stop, self.start)
        return self.stop - self.start

    @property
    def mid(self):
        return self.start + div_by_pow2(self.size)

    def __repr__(self):
        return (
            f"<{type(self).__name__} start={self.start} stop={self.stop} "
            f"left={self.left_visited} right={self.right_visited}>"
        )

    def left(self):
        return Task(self.start, self.mid)

    def right(self):
        return Task(self.mid, self.stop)


def fht2dt_non_rec(img: Image, sign: Sign) -> ADRTResult:
    """
    Same as fht2ds, but division is done in powers of 2
    """
    n = len(img)
    if n < 2:
        return ADRTResult(img, op_count=0)
    tasks = [Task(0, n)]
    task = tasks[-1]
    total_op_count = 0
    img = img[:]
    while True:
        if not task.left_visited:
            if task.size > 2:
                task = task.left()
                tasks.append(task)
            else:
                task.left_visited = True
            continue
        if not task.right_visited:
            if task.size > 2:
                task = task.right()
                tasks.append(task)
            else:
                task.right_visited = True
            continue
        if task.size > 1:
            img[task.start : task.stop], op_count = mergeHT(
                ADRTResult(img[task.start : task.mid], op_count=0),
                ADRTResult(img[task.mid : task.stop], op_count=0),
                sign=sign,
            )
            total_op_count += op_count
        tasks.pop(-1)
        if not tasks:
            break
        task = tasks[-1]  # parent task
        if not task.left_visited:
            task.left_visited = True
        elif not task.right_visited:
            task.right_visited = True
    return ADRTResult(img, op_count=total_op_count)


def mod(a: int, b: int):
    return a % b


def mergeHT(h0_res: ADRTResult, h1_res: ADRTResult, sign: Sign) -> ADRTResult:
    h0, h1 = h0_res.image, h1_res.image
    n0, m = len(h0), len(h0[0])
    n1 = len(h1)
    n = n0 + n1
    h: Image = [[]] * n
    r0 = (n0 - 1) / (n - 1)
    r1 = (n1 - 1) / (n - 1)
    for t in range(n):
        t0 = round(t * r0)  # use "common.round05" to match `fht2ids`
        t1 = round(t * r1)
        s = mod(sign * (t - t1), m)
        line = h1[t1]
        h[t] = add(h0[t0], line[s:] + line[:s])
    return ADRTResult(h, op_count=n * m + h0_res.op_count + h1_res.op_count)
