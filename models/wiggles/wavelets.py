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

    return start_times, end_times, identities, prototypes, N


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
def implicit_basis_to_C_format(basis):
    start_times, end_times, identities, prototypes, N = basis
    starray = np.asarray(start_times, dtype=np.int32)
    etarray = np.asarray(end_times, dtype=np.int32)
    idarray = np.asarray(identities, dtype=np.int32)

    n1 = len(prototypes)
    n2 = np.max([len(l) for l in prototypes])
    m = np.matrix(np.ones((n1, n2), dtype=np.float64)*np.nan)
    for i, p in enumerate(prototypes):
        m[i,:len(p)] = p

    return starray, etarray, idarray, m, N


def construct_implicit_basis_C(*args, **kwargs):
    basis = construct_implicit_basis(*args, **kwargs)
    return implicit_basis_to_C_format(basis)


def merge_implicit_bases(basis1, basis2):
    sts1, ets1, ids1, pts1, N1 = basis1
    sts2, ets2, ids2, pts2, N2 = basis2

    n_basis1 = len(sts1)
    n_basis2 = len(sts2)

    # assume there are no duplicates *within* the bases, so
    # each prototype can have at most one duplicate in the
    # other basis.
    dupes_i = set()
    dupes_j = set()
    id2_remap = dict()
    for j,pt2 in enumerate(pts2):
        id2_remap[j] = j+len(pts1)
        for i,pt1 in enumerate(pts1):
            if len(pt1)==len(pt2) and (pt1==pt2).all():
                dupes_i.add(i)
                dupes_j.add(j)
                id2_remap[j] = i
                break

    pts2 = [pt for j, pt in enumerate(pts2) if j not in dupes_j]
    ttt = [(st2, et2, id2_remap[id2]) for (st2, et2, id2) in zip(sts2, ets2, ids2) if id2_remap[id2] is not None]
    sts2, ets2, ids2 = zip(*ttt)


    # the joint basis starts with the first basis, truncating any elements
    # that would bleed over into the second basis and aren't explicit duplicates
    sts_joint = list(sts1)
    ets_joint = [et1 if id1 in dupes_i else min(et1, N1) for (et1, id1) in zip(ets1, ids1)]
    ids_joint = list(ids1)
    pts_joint = list(pts1) + list(pts2)

    # then we add the second basis, truncating any elements that would start
    # during the reign of the first basis. This involves starting in the
    # middle of an existing prototype, which we implement by creating a
    # new (truncated) prototype as necessary.
    for (st2, et2, id2) in zip(sts2, ets2, ids2):
        st2 += N1
        et2 += N1
        if st2 < N1:
            if id2 < len(pts1):
                # if this is a duplicate
                # and occurs at the overlap
                # point, skip it
                continue
            prefix = N1-st2
            pt = pts2[id2-len(pts1)]
            pt = pt[prefix:].copy()
            id2 = len(pts_joint)
            pts_joint.append(pt)
            st2 = N1
        sts_joint.append(st2)
        ets_joint.append(et2)
        ids_joint.append(id2)

    return sts_joint, ets_joint, ids_joint, pts_joint, N1+N2

def trim_basis(basis, elements, N=None):
    """
    Given an (implicit) basis representation, and a boolean array
    representing a subset of elements, return the basis containing
    only those elements.
    """
    sts, ets, ids, pts, oldN = basis
    elements = np.asarray(elements, dtype=bool)
    sts = np.asarray(sts)[elements]

    ets = np.asarray(ets)[elements]
    ids = np.asarray(ids)[elements]
    id_set = set(ids)
    id_map = dict()
    pts2 = []
    for i in range(len(pts)):
        if i in id_set:
            id_map[i] = len(pts2)
            pts2.append(pts[i])
    ids2 = [id_map[d] for d in ids]

    if N is None:
        # it's not trivial to determine the "true" end time of this basis
        # (i.e., time after which it stops being an orthogonal basis),
        # so we allow the caller to specify such a time if they know it.
        N = np.max(ets)

    return (sts, ets, ids2, pts2, N)

def construct_implicit_basis_preset(srate, preset_str, c_format=True):

    def joint_basis(len1=30, len2=300, smooth_levels=3):
        basis_smooth = construct_implicit_basis(srate, "db4_%.2f_%d_%d" % (srate, smooth_levels, len1), sort=False)
        basis_full = construct_implicit_basis(srate, "db4_%.2f_1_%d" % (srate, len2), sort=False)
        basis_joint = merge_implicit_bases(basis_smooth, basis_full)
        return basis_joint, basis_smooth, basis_full

    def split_joint_basis(min_repeatable_len=5, skip_levels=None, **kwargs):
        basis_joint, basis_smooth, basis_full = joint_basis(**kwargs)


        pts_smooth = basis_smooth[3]
        pt_lengths = sorted([len(pt) for pt in pts_smooth])
        if skip_levels is None:
            skip_lengths = [l for l in pt_lengths if l <= min_repeatable_len]
        else:
            skip_lengths = pt_lengths[:skip_levels]
        print pt_lengths
        print skip_lengths

        ids_smooth = basis_smooth[2]
        ids_joint = basis_joint[2]
        non_minimal_pts = set([j for j in range(len(pts_smooth)) if len(pts_smooth[j]) not in skip_lengths ])
        repeatable_elements = [(ids_smooth[j] in non_minimal_pts) for j in range(len(ids_smooth))]
        repeatable_elements += [False,] * (len(ids_joint)-len(ids_smooth))
        repeatable_elements = np.array(repeatable_elements, dtype=bool)
        N1 = basis_smooth[-1]
        return basis_joint, repeatable_elements, N1

    if preset_str=="wp1_repeatable":
        # at srate=5.0, we'd skip the first level, length 8
        min_repeatable_len = srate* 8.0/5.0
        basis_joint, repeatable_elements, N1 = split_joint_basis(len1=30, len2=300,
                                                                 min_repeatable_len=min_repeatable_len,
                                                                 smooth_levels=3)
        result = trim_basis(basis_joint, repeatable_elements, N=N1)
    elif preset_str=="wp1_nonrepeatable":
        min_repeatable_len = srate* 8.0/5.0
        basis_joint, repeatable_elements, N1 = split_joint_basis(len1=30, len2=300,
                                                                 min_repeatable_len=min_repeatable_len,
                                                                 smooth_levels=3)
        result = trim_basis(basis_joint, ~repeatable_elements, N=basis_joint[-1])

    if c_format:
        return implicit_basis_to_C_format(result)
    else:
        return result
