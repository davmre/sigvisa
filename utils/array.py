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

def index_to_time(idx, stime, srate):
    return idx / float(srate) + stime

def time_to_index(t, stime, srate):
    # we add the 1e-4 to fix numerical issues, so that floor()
    # doesn't round down if we get .99999995103... or similar.
    return int(np.floor((t - stime) * srate + 1e-4))

def time_to_index_offset(t, stime, srate):
    start = (t - stime) * srate + 1e-4
    start_idx = int(np.floor(start))
    offset = start - start_idx
    return start_idx, offset
