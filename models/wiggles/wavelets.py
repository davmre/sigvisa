import numpy as np
import pywt

def parse_wavelet_basis_str(s):
    ss = s.split("_")

    family, res, levels, len_s = ss
    res = float(res)
    levels=int(levels)
    len_s = float(len_s)
    return family, res, levels, len_s


def construct_basis_simple(N, family, pad, level=None):
    z = np.zeros((N,))
    cc = pywt.wavedec(z, family, pad, level=level)
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



def construct_implicit_basis_simple(N, family, pad, level=None):

    w = pywt.Wavelet(family)
    cc = pywt.wavedec(np.zeros((N,)), family, pad, level=level)

    level_sizes = [len(ccc) for ccc in cc]

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

    return start_times, end_times, identities, prototypes, level_sizes, N


def construct_full_basis(srate, wavelet_str=None, wavelet_family='db4', wiggle_len_s=30.0, decomp_levels=99):
    if wavelet_str is not None:
        # overwrite all the other params
        wavelet_family, _, decomp_levels, wiggle_len_s = parse_wavelet_basis_str(wavelet_str)

    N = int(wiggle_len_s * srate)
    basis = construct_basis_simple(N, wavelet_family, 'zpd', level=decomp_levels)
    return basis

def construct_full_basis_implicit(srate, wavelet_str=None, wavelet_family='db4', wiggle_len_s=30.0, decomp_levels=99, c_format=True):
    if wavelet_str is not None:
        # overwrite all the other params
        wavelet_family, _, decomp_levels, wiggle_len_s = parse_wavelet_basis_str(wavelet_str)

    N = int(wiggle_len_s * srate)
    basis = construct_implicit_basis_simple(N, wavelet_family, 'zpd', level=decomp_levels)
    if c_format:
        return implicit_basis_to_C_format(basis)
    else:
        return basis

def level_sizes(srate, wavelet_str):
    if wavelet_str is not None:
        # overwrite all the other params
        wavelet_family, _, decomp_levels, wiggle_len_s = parse_wavelet_basis_str(wavelet_str)

# construct an implicit representation of a wavelet basis, returned
# in the format required by the fast C++ implementation of
# the CompactSupportSSM
def implicit_basis_to_C_format(basis):
    start_times, end_times, identities, prototypes, levels, N = basis
    starray = np.asarray(start_times, dtype=np.int32)
    etarray = np.asarray(end_times, dtype=np.int32)
    idarray = np.asarray(identities, dtype=np.int32)

    n1 = len(prototypes)
    n2 = np.max([len(l) for l in prototypes])
    m = np.matrix(np.ones((n1, n2), dtype=np.float64)*np.nan)
    for i, p in enumerate(prototypes):
        m[i,:len(p)] = p

    return starray, etarray, idarray, m, levels, N

def implicit_to_explicit(start_times, end_times, identities, prototypes, levels, N):
    m = len(start_times)
    basis = []
    for i in range(m):
        st = max(start_times[i], 0)
        et = min(end_times[i], N)
        npts= et-st
        offset = st-start_times[i]

        # this is kinda awkward but makes autograd happy
        thisbasis = np.concatenate([np.zeros((st,)), prototypes[identities[i]][offset:offset+npts], np.zeros((N-et))])

        basis.append(thisbasis)
    basis = np.array(basis)
    return basis
