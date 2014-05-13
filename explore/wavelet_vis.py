from sigvisa.explore.doublets.xcorr_pairs import xcorr, xcorr_valid, extract_phase_window
from sigvisa.utils.geog import dist_km
from sigvisa import Sigvisa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pywt
import os



def load_signals(sta, basedir):
    s = np.load(os.path.join(basedir, "signals_Pn_%s.npy" % sta))
    evs = s[:, :7]
    signals = s[:,7:]
    return evs, signals


def xcorr_pairs(S1, S2, dists=None, dist_threshold=40, npts=None):
    xcorrs = np.zeros((S1.shape[0], S2.shape[0]))
    offsets = np.zeros((S1.shape[0], S2.shape[0]))
    if npts is None:
        npts = S1.shape[1]
    for (i, s1) in enumerate(S1):
        if i % 30 == 1:
            print "xc event %d" % i
        for (j, s2) in enumerate(S2):
            if j < i:
                continue
            if dist_threshold is not None and dists[i,j] > dist_threshold:
                continue
            xcmax, offset = xcorr(s1[:npts], s2[:npts])
            if np.isnan(xcmax): continue
            xcorrs[i,j] = xcmax
            offsets[i,j] = offset
    return xcorrs, offsets

"""
def xcorr_pairs(S1, S2, npts=None):
    xcorrs = np.zeros((S1.shape[0], S2.shape[0]))
    if npts is None:
        npts = S1.shape[1]
    for (i, s1) in enumerate(S1):
        if i % 30 == 1:
            print "xc event %d" % i
        for (j, s2) in enumerate(S2):
            if j < i:
                xcorrs[i,j] = xcorrs[j,i]
                continue
            xcmax, offset = xcorr(s1[:npts], s2[:npts])
            if np.isnan(xcmax): continue
            xcorrs[i,j] = xcmax
    return xcorrs
"""

def distance_pairs(X1, X2):
    dists = np.zeros((X1.shape[0], X2.shape[0]))
    for (i, x1) in enumerate(X1):
        for (j, x2) in enumerate(X2):
            dists[i,j] = dist_km((x1[0], x1[1]), (x2[0], x2[1]))
    return dists

def plot_aligned(s1, s2):
    xcmax, offset = xcorr(s1, s2)
    x = np.arange(0, len(s1))
    plot(x, s1/np.linalg.norm(s1, 2))
    plot(x + offset, s2/np.linalg.norm(s2, 2))


def signals_to_wavelets(S):
    wavelets = np.zeros((S.shape[0], 76))
    for (i, s) in enumerate(S):
        if np.std(s) == 0: continue
        s = (s - np.mean(s)) / (np.std(s) * np.sqrt(len(s)))
        coefs = pywt.wavedec(s[200:800], 'db8', mode='per')
        wavelets[i,:] = flatten_wavelet_coefs(coefs)
    return wavelets

def flatten_wavelet_coefs(coefs, levels=3):
    return np.concatenate([coefs[i] for i in range(levels)])

"""
def unflatten_wavelet_coefs_db9(cc):
    coefs = []

    for i in range(len(bounds)-1):
        if len(cc) >= bounds[i+1]:
            coefs.append(cc[bounds[i]:bounds[i+1]])
        else:
            coefs.append(np.zeros((bounds[i+1] - bounds[i])))
    return coefs
"""

def unflatten_square_index(i, nn):
    n = int(np.sqrt(nn))
    return i / n, i % n

def self_xc(s):
    base = s[-300:-100]
    xcmax, offset = xcorr_valid(s[:-300], base)
    return xcmax

def fourier_features(S):
    features = np.zeros((S.shape[0], 130))

    if S.shape[1] == 1200:
        for (i, s) in enumerate(S):
            ps = np.abs(np.fft.fft(s))**2
            f = ps[50:180]
            features[i,:] = f / np.linalg.norm(f, 2)
    elif S.shape[1] == 2400:
        for (i, s) in enumerate(S):
            ps = np.abs(np.fft.fft(s[::2]))**2
            f = ps[50:180]
            features[i,:] = f / np.linalg.norm(f, 2)
    return features

def wavelet_features(S, level=3):
    n = len(pywt.wavedec(S[0,:], 'db8', mode='per')[level])
    features = np.zeros((S.shape[0], n))

    for (i, s) in enumerate(S):
        c = pywt.wavedec(s, 'db8', mode='per')[level]
        features[i,:] = c / np.linalg.norm(c, 2)
    return features

def unaligned_wavelets(s, level):
    c = pywt.wavedec(s, 'db8', mode='per')[level]
    return c / np.linalg.norm(c, 2)

def aligned_wavelets(s, s_align, level, base_start, base_end):
    xcmax, offset = xcorr(s, s_align)
    return unaligned_wavelets(s[base_start+offset:base_end+offset], level=level)

def plot_pair(s1, s2, txt):
    fig = plt.figure(figsize=(15, 9))
    # plot aligned waveforms
    # and fourier psds
    # and unaligned wavelets at 3 levels
    # and aligned wavelets at 3 levels

    if len(s1) == 1200:
        base_start = 200
        base_end = 800
    elif len(s1) == 2400:
        base_start = 400
        base_end = 1600

    ax_waves = plt.subplot2grid((5,4), (0,0), colspan=4, rowspan=2)
    xcmax, offset = xcorr(s1, s2)
    x = np.arange(0, len(s1))
    ax_waves.plot(x, s1/np.linalg.norm(s1, 2))
    ax_waves.plot(x + offset, s2/np.linalg.norm(s2, 2))
    ax_waves.set_title("xc peak %.4f offset %.3fs %s" % (xcmax, offset/20.0, txt))

    ax_fourier = plt.subplot2grid((5,4), (2,0), rowspan=3, colspan=2)
    ff = fourier_features(np.vstack((s1, s2)))
    x = np.linspace(0.8,3.0, 130) # this is only approximate
    ax_fourier.plot(x, ff[0,:])
    ax_fourier.plot(x, ff[1,:])
    ax_fourier.set_title("fourier similarity %.4f" % np.dot(ff[0,:], ff[1,:]))

    ax_unaligned_wavelet0 = plt.subplot2grid((5,4), (2,2))
    w1 = unaligned_wavelets(s1[base_start:base_end], 0)
    w2 = unaligned_wavelets(s2[base_start:base_end], 0)
    ax_unaligned_wavelet0.plot(w1)
    ax_unaligned_wavelet0.plot(w2)
    ax_unaligned_wavelet0.set_title("unaligned a0: %.4f" % np.dot(w1, w2))

    ax_unaligned_wavelet1 = plt.subplot2grid((5,4), (3,2))
    w1 = unaligned_wavelets(s1[base_start:base_end], 1)
    w2 = unaligned_wavelets(s2[base_start:base_end], 1)
    ax_unaligned_wavelet1.plot(w1)
    ax_unaligned_wavelet1.plot(w2)
    ax_unaligned_wavelet1.set_title("unaligned d0: %.4f" % np.dot(w1, w2))

    ax_unaligned_wavelet2 = plt.subplot2grid((5,4), (4,2))
    w1 = unaligned_wavelets(s1[base_start:base_end], 2)
    w2 = unaligned_wavelets(s2[base_start:base_end], 2)
    ax_unaligned_wavelet2.plot(w1)
    ax_unaligned_wavelet2.plot(w2)
    ax_unaligned_wavelet2.set_title("unaligned d1: %.4f" % np.dot(w1, w2))


    ax_aligned_wavelet0 = plt.subplot2grid((5,4), (2,3))
    w1 = unaligned_wavelets(s1[base_start:base_end], 0)
    w2 = aligned_wavelets(s2, s1, 0, base_start=base_start, base_end=base_end)
    ax_aligned_wavelet0.plot(w1)
    ax_aligned_wavelet0.plot(w2)
    ax_aligned_wavelet0.set_title("xc-aligned a0: %.4f" % np.dot(w1, w2))


    ax_aligned_wavelet1 = plt.subplot2grid((5,4), (3,3))
    w1 = unaligned_wavelets(s1[base_start:base_end], 1)
    w2 = aligned_wavelets(s2, s1, 1, base_start=base_start, base_end=base_end)
    ax_aligned_wavelet1.plot(w1)
    ax_aligned_wavelet1.plot(w2)
    ax_aligned_wavelet1.set_title("xc-aligned d0: %.4f" % np.dot(w1, w2))


    ax_aligned_wavelet2 = plt.subplot2grid((5,4), (4,3))
    w1 = unaligned_wavelets(s1[base_start:base_end], 2)
    w2 = aligned_wavelets(s2, s1, 2, base_start=base_start, base_end=base_end)
    ax_aligned_wavelet2.plot(w1)
    ax_aligned_wavelet2.plot(w2)
    ax_aligned_wavelet2.set_title("xc-aligned d1: %.4f" % np.dot(w1, w2))
    plt.tight_layout()

def feature_dots(W1, W2, coef_start=0, coef_end=None):
    if coef_end is None:
        coef_end = W1.shape[1]
    dots =  np.zeros((W1.shape[0], W2.shape[0]))
    for (i, w1) in enumerate(W1):
        for (j, w2) in enumerate(W2):
            n1 = w1[coef_start:coef_end]
            #n1 = n1 - np.mean(n1)
            #n1 = n1 / np.linalg.norm(n1, 2)

            n2 = w2[coef_start:coef_end]
            #n2 = n2 - np.mean(n2)
            #n2 = n2 / np.linalg.norm(n2, 2)

            dots[i,j] = np.dot(n1, n2)

    return dots
