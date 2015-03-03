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

def construct_basis_simple(N, family, pad):
    z = np.zeros((N,))
    cc = pywt.wavedec(z, family, pad)
    nb = np.sum([len(l) for l in cc])
    basis = np.zeros((nb, N))
    k=0
    for i in range(len(cc)):
        for j in range(len(cc[i])):
            cc[i][j] = 1
            basis[k+j, :] = pywt.waverec(cc, family, pad)
            cc[i][j] = 0
        k += len(cc[i])
    return basis


def construct_implicit_basis_simple(N, family, pad):

    w = pywt.Wavelet(family)
    cc = pywt.wavedec(np.zeros((N,)), family, pad)

    start_times = []
    end_times = []
    prototypes = []
    identities = []
    for i,l in enumerate(cc):
        minus_level = len(cc)-i
        if i==0:
            minus_level = len(cc)-1
        increase = 2**minus_level


        # len at first level is w.dec_len
        # at second level, we cover w.dec_len coefs
        # the first of which covers w.dec_len spaces
        # and each of the rest adds another 2.
        # so it's wdl + (wdl-1)*2
        # at the third level, we cover wdl 2coefs
        # the first one of which covers wdl + (wdl-1)*2, i.e. the previous length
        # and each of the rest adds "increase"
        # so it's:
        # A = wdl
        # B = A + 2 * (wdl-1)
        # C = B + 4 * (wdl-1)
        # D = C + 8 * (wdl-1)
        # etc.
        # so the total (wdl-1) contribution is
        length = 1 + (w.dec_len -1) * np.sum([2**n for n in range(0, minus_level) ])
        #print "length at level", i, length

        base_start = (-w.dec_len+2)*np.sum([2**n for n in range(0, minus_level) ])
        start_j = int(np.ceil(-base_start / float(increase)))
        start_idx = base_start + start_j * increase
        #print "start_j", start_j, "from", -base_start / float(increase),  "idx", start_idx

        end_idx = start_idx+length
        l[start_j] = 1
        b = pywt.waverec(cc, family, pad)
        l[start_j] = 0
        prototypes.append(b[start_idx:end_idx])
        #print "extracting from", start_idx, "to", end_idx

        for j in range(len(l)):
            st = base_start + j * increase
            start_times.append(st)
            end_times.append(st+length)
            identities.append(i)

    return start_times, end_times, identities, prototypes


def construct_implicit_basis(srate, wavelet_str=None, wavelet_family='db4', wavelet_resolution_hz=0.5, wiggle_len_s=30.0, decomp_levels=99, sort=True):

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
    real_levels = len(cc)
    lastlen = len(cc[-1])

    signal_len = 2*(lastlen  - w.dec_len/2 + 1)
    while signal_len < wiggle_len_s * srate:
        cc.append(np.zeros((signal_len,)))
        signal_len = 2*(signal_len - w.dec_len/2 + 1)

    N = signal_len

    nb = np.sum([len(l) for l in cc[:real_levels]])
    start_times = []
    end_times = []
    prototypes = []
    identities = []
    for i,l in enumerate(cc[:real_levels]):
        minus_level = len(cc)-i
        if i==0:
            minus_level = len(cc)-1
        increase = 2**minus_level

        length = 1 + (w.dec_len -1) * np.sum([2**n for n in range(0, minus_level) ])
        base_start = (-w.dec_len+2)*np.sum([2**n for n in range(0, minus_level) ])
        start_j = int(np.ceil(-base_start / float(increase)))
        start_idx = base_start + start_j * increase
        end_idx = start_idx+length

        prototype = np.zeros((length,))
        prototype_coverage = np.zeros((length,))
        prototype_complete = False

        for j in range(len(l)):

            st = base_start + j * increase
            et = st+length
            start_times.append(st)
            end_times.append(et)
            identities.append(i)

            if not prototype_complete:

                l[j] = 1
                b = pywt.waverec(cc, wavelet_family, 'zpd')
                l[j] = 0

                # st and et tell us where we are in the signal.
                # if st is positive, we start at the beginning of the prototype,
                # otherwise w etsrat in the middle.
                # if et < N, we end at the end of the prototype, else we end in the middle
                pt_st = max(0, -st)
                pt_et = min(length, length - (et-N))
                prototype[pt_st:pt_et] = b[max(0, st): min(et, N)]
                prototype_coverage[pt_st:pt_et] = 1
                #print "j=%d: filling prototype[%d:%d] from b[%d:%d]" % (j, pt_st, pt_et, max(0, st), min(et, N))
                if np.sum(prototype_coverage) == length:
                    #print "prototype complete!"
                    prototype_complete=True

        prototypes.append(prototype)
        if len(prototypes[-1]) != length:
            import pdb; pdb.set_trace()
        assert(len(prototypes[-1]) == length)


    # return basis vectors in order of their starttimes. This means we
    # can increase the wiggle length without scrambling the earlier
    # vectors.
    if sort:
        p = sorted(np.arange(nb), key = lambda i : start_times[i])
        start_times = np.asarray(start_times)[p]
        end_times = np.asarray(end_times)[p]
        identities = np.asarray(identities)[p]

    return start_times, end_times, identities, prototypes, N

# construct an implicit representation of a wavelet basis, returned
# in the format required by the fast C++ implementation of
# the CompactSupportSSM
def construct_implicit_basis_C(*args, **kwargs):
    start_times, end_times, identities, prototypes, N = construct_implicit_basis(*args, **kwargs)

    starray = np.asarray(start_times, dtype=np.int32)
    etarray = np.asarray(end_times, dtype=np.int32)
    idarray = np.asarray(identities, dtype=np.int32)

    n1 = len(prototypes)
    n2 = np.max([len(l) for l in prototypes])
    m = np.matrix(np.ones((n1, n2), dtype=np.float64)*np.nan)
    for i, p in enumerate(prototypes):
        m[i,:len(p)] = p

    return starray, etarray, idarray, m, N
