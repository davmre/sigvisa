# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
import numpy as np

from sigvisa.utils import LogNormal

from sigvisa.database.dataset import *

SNR_STEP = 1
SNR_BINS = np.arange(0, 100, SNR_STEP)


def learn(param_filename, options, earthmodel, detections, leb_events,
          leb_evlist, false_dets):
    # learn the phase-SNR distribution for true detections
    true_bins = np.zeros((earthmodel.NumTimeDefPhases(), len(SNR_BINS)))
    for detlist in leb_evlist:
        for phase, detnum in detlist:
            snr = int(detections[detnum, DET_SNR_COL] // SNR_STEP)
            if snr >= len(SNR_BINS):
                snr = len(SNR_BINS) - 1
            true_bins[phase, snr] += 1

    # smooth the distribution by add one smoothing
    true_bins += 1.
    # normalize
    true_bins = (true_bins.T / true_bins.T.sum(axis=0)).T

    # REPEAT for false detections

    false_bins = np.zeros(len(SNR_BINS))
    # learn the false detections distribution
    for detnum in false_dets:
        snr = int(detections[detnum, DET_SNR_COL] // SNR_STEP)
        if snr >= len(SNR_BINS):
            snr = len(SNR_BINS) - 1
        false_bins[snr] += 1

    # smooth the distribution by add one smoothing
    false_bins += 1.
    # normalize
    false_bins /= false_bins.sum()

    np.set_printoptions(precision=2, threshold=10000)
    print "True phase -- SNR  distribution:"
    print true_bins
    print "False detection SNR distribution:"
    print false_bins
    np.set_printoptions()                 # restore defaults

    fp = open(param_filename, "w")
    print >> fp, SNR_STEP, len(SNR_BINS), earthmodel.NumTimeDefPhases()

    def print_arr(fp, arr):
        for x in arr:
            print >>fp, x,
        print >> fp

    print_arr(fp, false_bins)
    for phase in range(earthmodel.NumTimeDefPhases()):
        print_arr(fp, true_bins[phase])

    fp.close()

    if options.gui:
        plt.figure(figsize=(8, 4.8))
        if not options.type1:
            plt.title("SNR for true and false detections -- all sites")
        plt.plot(SNR_BINS, false_bins, linewidth=3, label="False")
        for phase in range(earthmodel.NumTimeDefPhases()):
            plt.plot(SNR_BINS, true_bins[phase], linewidth=3,
                     label=earthmodel.PhaseName(phase))
            if phase > 2:
                break
        plt.legend()
        plt.xlabel("SNR")
        plt.ylabel("Probability")
        if options.writefig is not None:
            basename = os.path.join(options.writefig, "ArrivalSNR")
            if options.type1:
                plt.savefig(basename + ".pdf")
            else:
                plt.savefig(basename + ".png")
