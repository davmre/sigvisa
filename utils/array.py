import numpy as np

def array_overlap(a, b, i):
    n1 = len(a)
    n2 = len(b)

    a_start = max(i, 0)
    a_end = max(a_start, min(n1, i+n2))

    b_start = max(-i, 0)
    b_end = max(b_start, min(n2, n1-i))

    assert(a_start >= 0 and a_start <= n1)
    assert(a_end >= 0 and a_end <= n1 and a_end >= a_start)
    assert(b_start >= 0 and b_start <= n1)
    assert(b_end >= 0 and b_end <= n1 and b_end >= b_start)
    return a[a_start:a_end], b[b_start:b_end]
