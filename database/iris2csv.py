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
# convert the IRIS event bulletin (in WEED format) into a CSV file
# with the following schema:
#
# create table isc_events (
#  eventid     int not null,
#  region      varchar(100) not null,
#  author      varchar(10) not null,
#  lon         double not null,
#  lat         double not null,
#  depth       double not null,
#  time        double not null,
#  mb          double
#  ndef        int,
#  nsta        int,
#  gap         int,
#  ml          float,
#
#  primary key (eventid, author),
#  index (time),
#  index (author, time)
# ) engine = myisam;
#
# Note: the WEED file has the following format
# catalog/origin-contributor, time, lat, lon, depth, code, regionid, magtype, mag [, magtype, mag]*
#
# data downloaded from http://www.iris.edu/SeismiQuery/sq-events.htm

import sys
import calendar


def main():
    if len(sys.argv) != 2:
        print "Error: Usage python iris2csv.py <IRIS-weed bulletin>"
        sys.exit(1)

    in_fname = sys.argv[1]
    out_fname = in_fname + ".csv"

    in_fp = open(in_fname)
    out_fp = open(out_fname, "w")

    print "Converting %s to %s" % (in_fname, out_fname)

    print >> out_fp, "eventid,region,author,lon,lat,depth,time,mb,ndef,nsta,"\
        "gap,ml"

    for line in in_fp:
        line = line.rstrip()
        fields = line.split(",")

        datestamp, timestamp = fields[1].split()
        year, mon, day = [int(x) for x in datestamp.split("/")]
        hr, mi, sec = [int(float(x)) for x in timestamp.split(":")]
        fracsec = float(timestamp.split(":")[2]) - sec

        tim = calendar.timegm((year, mon, day, hr, mi, sec)) + fracsec

        latitude, longitude, depth \
            = float(fields[2]), float(fields[3]), float(fields[4])

        mb, ml = -999, -999
        for magtype, mag in zip(fields[7::2], fields[8::2]):
            magtype = magtype.strip().lower()
            if magtype == "mb":
                mb = float(mag)
            elif magtype == "ml":
                ml = float(mag)

        eventid = int(tim)
        author, region, ndef, nsta, gap = "IRIS", "", -999, -999, -999
        print >> out_fp, '%s,"%s",%s,%f,%f,%f,%f,%f,%d,%d,%d,%f' \
            % (eventid, region, author, longitude, latitude, depth,
               tim, mb, ndef, nsta, gap, ml)

    in_fp.close()
    out_fp.close()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except:
        import pdb
        import traceback
        import sys
        traceback.print_exc(file=sys.stdout)
        pdb.post_mortem(sys.exc_traceback)
        raise
