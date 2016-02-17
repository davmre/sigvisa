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

from sigvisa.database.dataset import EV_LON_COL, EV_LAT_COL, EV_DEPTH_COL, EV_MB_COL,\
    EV_TIME_COL

from sigvisa.utils.geog import dist_deg, dist_km

import mwmatching

DELTA_TIME = 50                         # in seconds

def find_matching(gold_events, guess_events, max_delta_deg=1.0):
    """
    We want a max cardinality min cost matching.
    Returns a list of pairs of gold,guess indices
    """
    edges = []
    for goldnum, gold in enumerate(gold_events):
        for guessnum, guess in enumerate(guess_events):
            if ((abs(gold[EV_TIME_COL] - guess[EV_TIME_COL]) <= DELTA_TIME)
                    and (dist_deg(gold[:2], guess[:2]) <= max_delta_deg)):
                edges.append((goldnum, len(gold_events) + guessnum,
                              -dist_deg(gold[:2], guess[:2])))

    mat = mwmatching.maxWeightMatching(edges, maxcardinality=True)
    indices = []
    for i in range(len(gold_events)):
        if i < len(mat) and mat[i] >= 0:
            assert(mat[i] >= len(gold_events))
            indices.append((i, mat[i] - len(gold_events)))

    return indices


def find_true_false_guess(gold_events, guess_events):
    mat = find_matching(gold_events, guess_events)
    true = [j for (i, j) in mat]
    true_set = set(true)
    false = [i for i in range(len(guess_events)) if i not in true_set]

    return true, false, mat


def find_unmatched(gold_events, guess_events):
    mat = find_matching(gold_events, guess_events)
    mat_gold = set()
    for i, j in mat:
        mat_gold.add(i)
    return [x for x in range(len(gold_events)) if x not in mat_gold]


def find_matched(gold_events, guess_events):
    return list(set(i for (i, j) in find_matching(gold_events, guess_events)))


def f1_and_error(gold_events, guess_events, max_delta_deg=1.0):
    indices = find_matching(gold_events, guess_events, max_delta_deg=max_delta_deg)
    # compute precision
    if len(guess_events):
        p = 100. * float(len(indices)) / len(guess_events)
    else:
        p = 100.
    # compute recall
    if len(gold_events):
        r = 100. * float(len(indices)) / len(gold_events)
    else:
        r = 100.
    # compute f1
    if p == 0 or r == 0:
        f = 0.
    else:
        f = 2 * p * r / (p + r)
    # compute avg error
    if len(indices):
        errs = np.array([dist_km((gold_events[i, EV_LON_COL],
                                  gold_events[i, EV_LAT_COL]),
                       (guess_events[j, EV_LON_COL],
                        guess_events[j, EV_LAT_COL]))
            for (i, j) in indices])
        err = (np.average(errs), np.std(errs))
    else:
        err = (0., 0.)

    return f, p, r, err

