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

from sigvisa.database.dataset import *


def learn(param_fname, options, leb_events):
    # Gutenberg Richter dictates that the there are 10 times as many events
    # of magnitude >= k than are >= k+1
    mb_rate = np.log(10)

    fp = open(param_fname, "w")

    print >>fp, "%f %f" % (MIN_MAGNITUDE, mb_rate)

    fp.close()

    if options.gui:
        mbs = []

        # events with the minimum mb actually value have unknown mb, best to
        # leave them out for estimation
        for mb in leb_events[:, EV_MB_COL]:
            if mb > MIN_MAGNITUDE:
                mbs.append(mb)

        mbs = np.array(mbs)

        plt.figure(figsize=(8, 4.8))
        if not options.type1:
            plt.title("Event mb")
        plt.xlim(MIN_MAGNITUDE, MAX_MAGNITUDE)
        xpts = np.arange(MIN_MAGNITUDE, MAX_MAGNITUDE, .1)
        plt.hist(mbs, xpts, facecolor="blue", edgecolor="none", normed=True,
                 label="data", alpha=0.5)
        plt.plot(xpts, [mb_rate * np.exp(-mb_rate * (x - MIN_MAGNITUDE))
                        for x in xpts], color="blue", label="data")
        plt.legend(loc="upper left")
        if options.type1:
            plt.xlabel(r'$m_{b}$')
        else:
            plt.xlabel('mb')
        plt.ylabel("probability density")
        # save the figure
        if options.writefig is not None:
            basename = os.path.join(options.writefig, "EventMagPrior")
            if options.type1:
                plt.savefig(basename + ".pdf")
            else:
                plt.savefig(basename + ".png")
