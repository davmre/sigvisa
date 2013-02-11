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
import csv
import numpy as np
from sigvisa.database.dataset import *
import sigvisa.utils.GMM
import sigvisa.utils.LinearModel
import math

NUM_FEATURES = 5
FEATURE_NAMES = ["(intercept)", "mb", "depth", "ttime", "ttime0"]


def extract_features(mb, depth, ttime):
    return (1.0, mb, depth, ttime, np.exp(-ttime / 50.))

NUM_PRIOR = 100                         # number of prior points


def gtf(val, m, s):
    return math.exp(- float(val - m) ** 2 / (2.0 * float(s) ** 2)) \
        / math.sqrt(2.0 * math.pi * float(s) ** 2)


def print_2gmm(wts, means, stds):
    print "w1=%.2f m1=%.2f s1=%.2f w2=%.2f m2=%.2f s2=%.2f"\
          % (wts[0], means[0], stds[0],
             wts[1], means[1], stds[1])


def print_list(fp, list):
    for x in list:
        print >>fp, x,


def predict_amp_model(coeffs, mb, depth, ttime):
    return coeffs[0] + coeffs[1] * mb + coeffs[2] * depth \
        + coeffs[3] * ttime + coeffs[4] * np.exp(-ttime / 50.)


def learn(param_filename, options, earthmodel, detections, leb_events,
          leb_evlist, false_dets):
    # first the false detections
    false_logamps = []
    site_false_logamps = dict((sitenum, []) for sitenum in
                              range(earthmodel.NumSites()))
    for detnum in false_dets:

        if -1 == detections[detnum, DET_AMP_COL]:
            continue

        sitenum = int(detections[detnum, DET_SITE_COL])
        datum = np.log(detections[detnum, DET_AMP_COL])

        false_logamps.append(datum)
        site_false_logamps[sitenum].append(datum)

    # sample some points from the overall false detection empirical distribution
    false_prior = [false_logamps[np.random.randint(len(false_logamps))] for
                   i in range(NUM_PRIOR)]

    site_false_params = []
    print "False Arrival Amplitude:"
    for sitenum in range(earthmodel.NumSites()):
        data = site_false_logamps[sitenum] + false_prior
        wts, means, stds = utils.GMM.estimate(2, data)

        site_false_params.append((wts, means, stds))

        if options.verbose:
            print "[%d]: False" % (sitenum,),
            print_2gmm(wts, means, stds)

    # next, the set of true detections
    # create a dataset for each phase and site
    phase_site_data = [[([], []) for site in xrange(earthmodel.NumSites())]
                       for phase in xrange(earthmodel.NumTimeDefPhases())]

    for evnum, detlist in enumerate(leb_evlist):
        for phase, detnum in detlist:

            # -1 => amplitude not observed
            if -1 == detections[detnum, DET_AMP_COL]:
                continue

            sitenum = int(detections[detnum, DET_SITE_COL])

            # compute the predictors
            pred = extract_features(leb_events[evnum, EV_MB_COL],
                                    leb_events[evnum, EV_DEPTH_COL],
                                    detections[detnum, DET_TIME_COL]
                                    - leb_events[evnum, EV_TIME_COL])
            data = phase_site_data[phase][sitenum]
            data[0].append(pred)
            data[1].append(np.log(detections[detnum, DET_AMP_COL]))

    # convert each phase-site's data into a matrix
    phase_site_data = [[(np.array(phase_site_data[phase][site][0]),
                         np.array(phase_site_data[phase][site][1]))
                        for site in xrange(earthmodel.NumSites())]
                       for phase in xrange(earthmodel.NumTimeDefPhases())]

    phase_site_coeffs = []
    phase_site_sigma = []
    print "Arrival Amplitude"
    for phase in xrange(earthmodel.NumTimeDefPhases()):
        print earthmodel.PhaseName(phase)
        site_coeffs, site_sigma, mean, sigma, beta\
            = sigvisa.utils.LinearModel.hier_linearreg(phase_site_data[phase])
        print zip(mean, sigma), beta
        phase_site_coeffs.append(site_coeffs)
        phase_site_sigma.append(site_sigma)

    # write out the arrival parameters
    fp = open(param_filename, "w")
    print >>fp, earthmodel.NumSites(), earthmodel.NumTimeDefPhases()

    # for each site
    for siteid in xrange(earthmodel.NumSites()):
        # first write the false arrival parameters
        wts, means, stds = site_false_params[siteid]
        print_list(fp, wts)
        print_list(fp, means)
        print_list(fp, stds)
        print >> fp
        # then the true phase arrival parameters
        for phaseid in xrange(earthmodel.NumTimeDefPhases()):
            print_list(fp, list(phase_site_coeffs[phaseid][siteid])
                       + [phase_site_sigma[phaseid][siteid]])
            print >> fp
    fp.close()

    if options.datadir:
        fname = os.path.join(options.datadir, "ArrivalAmplitude.csv")
        writer = csv.writer(open(fname, "wb"))
        writer.writerow(["PHASEID", "SITEID"] + FEATURE_NAMES + ["LOGAMP"])

        for phaseid in xrange(earthmodel.NumTimeDefPhases()):
            for siteid in xrange(earthmodel.NumSites()):
                data, samples = phase_site_data[phaseid][siteid]
                for d, s in zip(data, samples):
                    writer.writerow([phaseid, siteid] + list(d) + [s])
        # close the file
        del writer

    if not options.gui:
        return

    # learn the overall false detection model (for all sites)
    false_wts, false_means, false_stds = utils.GMM.estimate(2, false_logamps)

    if options.verbose:
        print "Overall False log(Amp):",
        print_2gmm(false_wts, false_means, false_stds)

    STEP = .1
    bins = np.arange(-7, 8, STEP)
    plt.figure(figsize=(8, 4.8))
    if not options.type1:
        plt.title("log(amp) for false detections -- all sites")
    plt.hist(false_logamps, bins, label="data", alpha=.5)
    plt.plot(bins, [utils.GMM.evaluate(false_wts, false_means, false_stds,
                                       x + STEP / 2)
                    * STEP * len(false_logamps) for x in bins], label="model",
             linewidth=3, color="black")

    plt.xlabel("log(amp)")
    plt.ylabel("frequency")
    plt.legend(loc="upper left")
    if options.writefig is not None:
        basename = os.path.join(options.writefig,
                                "ArrivalAmplitudeFalseAllSites")
        if options.type1:
            plt.savefig(basename + ".pdf")
        else:
            plt.savefig(basename + ".png")

    # visualize some of the site-specific models
    for sitenum in range(earthmodel.NumSites()):

        data = site_false_logamps[sitenum] + false_prior
        wts, means, stds = site_false_params[sitenum]

        if sitenum in [6, 113]:
            plt.figure(figsize=(8, 4.8))
            if not options.type1:
                plt.title("log(amp) for false detections -- %d" % sitenum)
            plt.hist(data, bins, label="data", alpha=.5)
            plt.plot(bins, [utils.GMM.evaluate(wts, means, stds, x + STEP / 2)
                            * STEP * len(data) for x in bins], label="model",
                     linewidth=3, color="black")
            plt.xlabel("log(amp)")
            plt.ylabel("frequency")
            plt.legend(loc="upper left")
            if options.writefig is not None:
                basename = os.path.join(options.writefig, "ArrivalAmplitudeFalseSite%d"
                                        % sitenum)
                if options.type1:
                    plt.savefig(basename + ".pdf")
                else:
                    plt.savefig(basename + ".png")


def create_featureset(earthmodel, start_time, end_time, detections, leb_events,
                      leb_evlist, sel3_events, sel3_evlist, site_up, sites,
                      phasenames, phasetimedef):

    phase_data = [([[] for _ in FEATURE_NAMES], [])
                  for phaseid in xrange(phasetimedef.sum())]

    for evnum, detlist in enumerate(leb_evlist):
        for phase, detnum in detlist:

            # -1 => amplitude not observed
            if -1 == detections[detnum, DET_AMP_COL]:
                continue

            sitenum = int(detections[detnum, DET_SITE_COL])
            if sitenum != 6:
                continue

            # compute the predictors
            pred = extract_features(leb_events[evnum, EV_MB_COL],
                                    leb_events[evnum, EV_DEPTH_COL],
                                    detections[detnum, DET_TIME_COL]
                                    - leb_events[evnum, EV_TIME_COL])
            data = phase_data[phase]
            for f in range(len(FEATURE_NAMES)):
                data[0][f].append(pred[f])
            data[1].append(np.log(detections[detnum, DET_AMP_COL]))

    return phase_data


def test_model(earthmodel, train, test):
    train_feat = create_featureset(earthmodel, *train)
    test_feat = create_featureset(earthmodel, *test)
    # we will evaluate with increasing number of features
    for numfeatures in range(1, len(FEATURE_NAMES) + 1):
        print "Evaluating features:", FEATURE_NAMES[:numfeatures]

        # train a model for each phase
        phase_model = []
        for phaseid, (predictors, output) in enumerate(train_feat):
            if len(output) < 2:
                phase_model.append(None)
                continue

            model = sigvisa.utils.LinearModel.LinearModel("Amp", FEATURE_NAMES[:numfeatures],
                                                          predictors[:numfeatures], output,
                                                          verbose=False)

            phase_model.append(tuple(model.coeffs) + (model.std,))

        # now predict on the test data
        totcnt = 0
        tot_logprob = 0.
        for phaseid, (predictors, output) in enumerate(test_feat):
            model = phase_model[phaseid]
            if model is None:
                continue

            for idx in range(len(output)):
                guess = sum(predictors[f][idx] * model[f]
                            for f in xrange(numfeatures))
                logprob = - 0.5 * np.log(2 * np.pi) - np.log(model[-1]) \
                          - 0.5 * (guess - output[idx]) ** 2 / model[-1] ** 2

                totcnt += 1
                tot_logprob += logprob

        print "Avg. log likelihood", tot_logprob / totcnt
