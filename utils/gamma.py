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
# a gamma distribution with location, shape, and scale
#
import numpy as np


def estimate(vals):

    location = min(vals) - .001

    cnt, tot, tot_log = 0, 0., 0.

    for val in vals:
        val -= location

        cnt += 1
        tot += val

        tot_log += np.log(val)

    avg = tot / cnt
    avg_log = tot_log / cnt

    s = np.log(avg) - avg_log

    assert s > 0

    # http://en.wikipedia.org/wiki/Gamma_distribution#Parameter_estimation
    shape = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12. * s)

    scale = avg / shape

    prev_loglike = (shape - 1) * tot_log - tot / scale - cnt * loggamma(shape) \
        - cnt * shape * np.log(scale)

    prev_params = (location, shape, scale)

    # Newton-Raphson updates
    while True:
        shape = shape - (np.log(shape) - digamma(shape) - s) \
            / (1.0 / shape - trigamma(shape))
        scale = avg / shape
        loglike = (shape - 1) * tot_log - tot / scale - cnt * loggamma(shape) \
            - cnt * shape * np.log(scale)

        params = (location, shape, scale)

        if np.isnan(loglike) or np.isinf(loglike) or loglike < prev_loglike:
            return prev_params

        elif abs(prev_loglike - loglike) < 1e-6:
            return params

        else:
            prev_loglike, prev_params = loglike, params


def digamma(k):
    return (loggamma(k + .001) - loggamma(k)) / .001

    # the following approximation from Wikipedia is very bad!
    if k < 8:
        return digamma(k + 1.) - 1. / k
    else:
        return np.log(k) - (1. + (1. - (1. / 10 - 1. / (21. * k ** 2)) / k ** 2) / (6. * k)) / (.2 * k)


def trigamma(k):
    return (digamma(k + .001) - digamma(k)) / .001

    # the following approximation from Wikipedia is very bad!
    if k < 8:
        return trigamma(k + 1.) + 1. / k ** 2
    else:
        return (1. + (1. + (1. - (1. / 5 - 1. / (7. * k ** 2)) / k ** 2) / (3. * k)) / (2. * k)) / k


def loggamma(num):
    return (num - .5) * np.log(num) - num + .5 * np.log(2 * np.pi)


def ldensity(params, val):
    location, shape, scale = params

    if val < location:
        val = 0
    else:
        val -= location

    assert shape > 0 and scale > 0

    return (shape - 1) * np.log(val) - val / scale - loggamma(shape) \
        - shape * np.log(scale)


def ldensity_list(params, vals):
    location, shape, scale = params

    newvals = []
    for val in vals:
        if val < location:
            val = 0
        else:
            val -= location

        newvals.append(val)

    assert shape > 0 and scale > 0
    sum_v = sum(newvals)
    sum_logv = sum(np.log(v) for v in newvals)

    return (shape - 1) * sum_logv - sum_v / scale - len(vals) * loggamma(shape) \
        - len(vals) * shape * np.log(scale)
