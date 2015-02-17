import numpy as np
import pywt

def parse_wavelet_basis_str(s):
    family, res, levels, len_s = s.split("_")
    res = float(res)
    levels=int(levels)
    len_s = float(len_s)
    return family, res, levels, len_s

def construct_wavelet_basis(srate, wavelet_str=None, wavelet_family='db4', wavelet_resolution_hz=0.5, wiggle_len_s=30.0, decomp_levels=99, sort=True):

    if wavelet_str is not None:
        # overwrite all the other params
        wavelet_family, wavelet_resolution_hz, decomp_levels, wiggle_len_s = parse_wavelet_basis_str(wavelet_str)


    N_base = int(wiggle_len_s * wavelet_resolution_hz)
    w = pywt.Wavelet(wavelet_family)
    levels = min(decomp_levels,
                 pywt.dwt_max_level(data_len=N_base, filter_len=w.dec_len))


    if wavelet_resolution_hz > srate:
        raise Exception("cannot request wavelet basis with higher resolution than the signal!")


    # while the total reconstruction length is less than desired, append zeros
    cc = pywt.wavedec(np.zeros(N_base,), wavelet_family, 'zpd', level=levels)
    print "orig", [len(l) for l in cc]
    real_levels = len(cc)
    lastlen = len(cc[-1])


    signal_len = 2*(lastlen  - w.dec_len/2 + 1)
    while signal_len < wiggle_len_s * srate:
        cc.append(np.zeros((signal_len,)))
        print "appending", 2*(signal_len - w.dec_len/2 + 1), "from", signal_len
        signal_len = 2*(signal_len - w.dec_len/2 + 1)

    print "padded", [len(l) for l in cc]
    N = signal_len

    nb = np.sum([len(l) for l in cc[:real_levels]])
    basis = np.empty((nb, N))
    k = 0
    start_times = []
    for i,l in enumerate(cc[:real_levels]):
        minus_level = len(cc)-i
        if i==0:
            minus_level = len(cc)-1
        increase = 2**minus_level
        base_start = (-w.dec_len+2)*np.sum([2**n for n in range(0, minus_level) ])
        for j in range(len(l)):
            cc[i][j] = 1.0
            basis[k+j, :] = pywt.waverec(cc, wavelet_family, 'zpd')
            cc[i][j] = 0.0
            st = base_start + j * increase
            start_times.append(st)
        k += j+1

    # return basis vectors in order of their starttimes. This means we
    # can increase the wiggle length without scrambling the earlier
    # vectors.
    #sorted_basis = np.array(sorted(, key = lambda b : np.min(np.arange(N)[np.abs(b) > 0])))

    if sort:
        p = sorted(np.arange(nb), key = lambda i : start_times[i])
        basis = basis[p,:]

    #sorted_basis = (1.0/np.max(np.abs(sorted_basis), axis=1)).reshape((-1,1)) * sorted_basis

    print "constructed wavelet basis with %d params for signal of length %d" % (basis.shape[0], N)

    return basis
