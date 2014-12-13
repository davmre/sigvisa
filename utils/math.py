import numpy as np

def safe_log(x, default=float('-inf')):
    # x is a scalar float
    if x <= 0:
        return default
    else:
        return np.log(x)

def safe_log_vec(x, default = float('-inf')):
    good_idx = x>0
    logx_good = np.log(x[good_idx])
    logx = default * np.ones(x.shape)
    logx[good_idx] = logx_good
    return logx
