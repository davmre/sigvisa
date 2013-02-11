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
# prunes events which are within a space-time ball of better events
import os
import sys
import numpy as np
from optparse import OptionParser

from sigvisa.database.dataset import *
from analyze import suppress_duplicates
import sigvisa.database.db


def main():
    parser = OptionParser()
    parser.add_option("-i", "--runid", dest="runid", default=None,
                      type="int",
                      help="the run-identifier to prune (last runid)")

    (options, args) = parser.parse_args()

    conn = database.db.connect()
    cursor = conn.cursor()

    if options.runid is None:
        cursor.execute("select max(runid) from visa_run")
        options.runid, = cursor.fetchone()

    print "RUNID %d:" % options.runid,

    cursor.execute("select run_start, run_end, data_start, data_end, descrip, "
                   "numsamples, window, step from visa_run where runid=%d" %
                   options.runid)

    run_start, run_end, data_start, data_end, descrip, numsamples, window, step\
        = cursor.fetchone()

    if data_end is None:
        print "NO RESULTS"
        return

    events, orid2num = read_events(cursor, data_start, data_end,
                                   "visa", options.runid)

    cursor.execute("select orid, score from visa_origin where runid=%d" %
                   (options.runid,))

    evscores = dict(cursor.fetchall())

    new_events, new_orid2num = suppress_duplicates(events, evscores)

    print "%d events, %d will be pruned" % (len(events),
                                            len(events) - len(new_events))

    for orid in orid2num.iterkeys():
        if orid not in new_orid2num:
            cursor.execute("delete from visa_origin where runid=%d and orid=%d"
                           % (options.runid, orid))
            cursor.execute("delete from visa_assoc where runid=%d and orid=%d"
                           % (options.runid, orid))
    conn.commit()

if __name__ == "__main__":
    main()
