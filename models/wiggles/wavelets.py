import numpy as np
import pywt

def construct_wavelet_basis(srate, wavelet_family='db4', wavelet_resolution_hz=0.5, wiggle_len_s=30.0):
    N = int(wiggle_len_s * srate)

    if wavelet_resolution_hz > srate:
        raise Exception("cannot request wavelet basis with higher resolution than the signal!")
    target_coefs = N * (wavelet_resolution_hz / float(srate))


    cc = pywt.wavedec(np.zeros(N,), wavelet_family, 'zpd')
    nb = np.sum([len(l) for l in cc])
    basis = np.empty((nb, N))
    k = 0
    for i,l in enumerate(cc):
        if k > target_coefs: break
        for j in range(len(l)):
            cc[i][j] = 1.0
            basis[k+j, :] = pywt.waverec(cc, wavelet_family, 'zpd')
            cc[i][j] = 0.0
        k += len(l)

    print "constructed wavelet basis with %d params for signal of length %d" % (k, N)
    small_basis = np.zeros((k, N))
    small_basis[:, :] = basis[:k, :]
    del basis
    return small_basis
