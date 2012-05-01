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
"""
creates a dataset from the IDC database
Usage: python idcexport.py <connect-string> <label> <start-time> <end-time> <data.tar>

The various tables in the dataset are
o. README.txt
   A text summary of the dataset.
o. dataset.csv
  - label
  - start_time
  - end_time
o. site.csv
  - sta
  - lon
  - lat
  - elev
  - isarray
o. phase.csv
  - phase
  - istimedef
o. idcx_arrival.csv & leb_arrival.csv
  - sta
  - arid
  - time
  - deltime
  - azimuth
  - delaz
  - slo
  - delslo
  - SNR
  - iphase
  - amplitude
  - period
o. leb_assoc.csv & sel3_assoc.csv
  - orid
  - arid
  - phase
  - sta
o. leb_origin.csv & sel3_origin.csv
  - lon
  - lat
  - depth
  - time
  - mb
  - orid

Example:

To extract the Tohoku seismic data from vDEC:

March 11 to March 13 2012
> python idcexport.py nimar/nimar@irdb tohoku-test 1299801600 1299974400 tohoku-test.tar --schemasuffix "_ref."

Dec 1 2010 to March 11 2012
> python idcexport.py nimar/nimar@irdb tohoku-train 1291161600 1299801600 tohoku-train.tar --schemasuffix "_ref."

"""
import sys, os, tarfile, csv
from datetime import datetime
import cx_Oracle
from optparse import OptionParser

TEMPNAME = "tempsave.csv.%d" % os.getpid()

PHASES = (
  ("P",   1),
  ("Pn",  1),
  ("PKP", 1),   
  ("Sn",  1),    
  ("S",   1),     
  ("PKPbc", 1), 
  ("PcP", 1),   
  ("pP",  1),
  ("Lg",  1),
  ("PKPab", 1),
  ("ScP", 1),
  ("PKKPbc", 1),
  ("Pg",  1),
  ("Rg",  1),
  ("tx",  0),
  ("Sx",  0),
  ("Px",  0),
  ("N",   0))

TIMEDEFPHASESET = set(x[0] for x in PHASES if x[1])
ALLPHASESET = set(x[0] for x in PHASES)

def export_query(cursor, query, tar, destfile, filterconds=None,
                 returnrows=False):
  if filterconds is None:
    filterconds = []
  #print query
  output = csv.writer(open(TEMPNAME, "wb"), quoting=csv.QUOTE_NONNUMERIC)
  cursor.execute(query)
  output.writerow([x[0] for x in cursor.description])
  rowcnt = 0
  rows = []
  while True:
    row = cursor.fetchone()
    if row is None:
      break
    
    # check all the filter conditions
    if sum(int(row[colnum] in colset) for colnum, colset in filterconds) \
       != len(filterconds):
      continue
    
    output.writerow(row)

    if returnrows:
      rows.append(row)
    else:
      rowcnt += 1
      
  del output
  tar.add(name = TEMPNAME, arcname=destfile)

  if returnrows:
    return rows
  else:
    return rowcnt

def main():
  #
  # parse arguments
  #
  parser = OptionParser()
  parser.set_usage("Usage: python %prog [options] <connect-string> <label> "
                   "<start-time> <end-time> <output.tar>")
  parser.add_option("--network", dest="network", default="SEISMIC",
                    help = "export data from specified network (SEISMIC)",
                    metavar="NAME")
  parser.add_option("--schemasuffix", dest="schemasuffix", default=".",
                    help ="suffix after schema e.g. '_ref.' (defaults to '.')",
                    metavar="suffix")
  (options, args) = parser.parse_args()
  if len(args) != 5:
    parser.print_help()
    sys.exit(1)

  connstr, label, start_time, end_time, output_name \
           = args[0], args[1], float(args[2]), float(args[3]), args[4]

  start_dt = datetime.utcfromtimestamp(start_time)
  start_yday = start_dt.year * 1000 + start_dt.timetuple().tm_yday
  end_dt = datetime.utcfromtimestamp(end_time)
  end_yday = end_dt.year * 1000 + end_dt.timetuple().tm_yday
  print "Exporting data from", start_dt, "to", end_dt
  
  # connect to the database
  dbconn = cx_Oracle.connect(connstr)
  cursor = dbconn.cursor()
  
  # open the tar file
  tar = tarfile.open(name=output_name, mode="w")

  # dataset.csv
  output = csv.writer(open(TEMPNAME, "wb"), quoting=csv.QUOTE_NONNUMERIC)
  output.writerow(["label", "start_time", "end_time"])
  output.writerow([label, start_time, end_time])
  del output
  tar.add(name = TEMPNAME, arcname="dataset.csv")

  # we want all the sites in the network which were on for the entire
  # duration of this dataset and they detected some arrival
  sitedata = export_query(cursor, "select site.sta, lon, lat, elev, "
               "decode(statype, 'ar', 1, 0) isarray from static%ssite site, "
               "static%saffiliation aff where aff.net='%s' and "
               "aff.sta=site.sta and (site.ondate < %d) and "
               "((site.offdate=-1) or (site.offdate > %d)) and exists "
               "(select 1 from idcx%sarrival arr where arr.sta=site.sta "
               " and arr.time between %f and %f) order by 1"
               % (options.schemasuffix, options.schemasuffix,
                  options.network, start_yday, end_yday, options.schemasuffix,
                  start_time, end_time)
              , tar, "site.csv", returnrows=True)
  staset = set(x[0] for x in sitedata)

  output = csv.writer(open(TEMPNAME, "wb"), quoting=csv.QUOTE_NONNUMERIC)
  output.writerow(["phase", "istimedef"])
  for row in PHASES:
    output.writerow(row)
  del output
  tar.add(name = TEMPNAME, arcname="phase.csv")
  
  numlebori = export_query(cursor,
                           "select lon, lat, depth, time, mb, orid from "
                           "leb%sorigin where time between %d and %d "
                           "order by time"
                           % (options.schemasuffix, start_time, end_time),
                           tar, "leb_origin.csv")
  
  numlebass = export_query(cursor,
                        "select orid, arid, phase, sta from leb%sassoc join "
                        "leb%sorigin ori using (orid) join leb%sarrival arr "
                        "using (arid, sta) where ori.time between "
                        "%d and %d order by ori.time, arr.time"
                        % (options.schemasuffix, options.schemasuffix,
                           options.schemasuffix, start_time, end_time),
                        tar, "leb_assoc.csv",
                        [(2, TIMEDEFPHASESET), (3,staset)])
  
  numsel3ori = export_query(cursor,
                           "select lon, lat, depth, time, mb, orid from "
                           "sel3%sorigin where time between %d and %d "
                           "order by time"
                           % (options.schemasuffix, start_time, end_time),
                           tar, "sel3_origin.csv")

  numsel3ass = export_query(cursor,
                         "select orid, arid, phase, sta from sel3%sassoc join "
                            "sel3%sorigin ori using (orid) join idcx%sarrival "
                            " arr using (arid, sta) where ori.time between "
                            "%d and %d order by ori.time, arr.time"
                            % (options.schemasuffix, options.schemasuffix,
                               options.schemasuffix, start_time, end_time),
                            tar, "sel3_assoc.csv",
                            [(2, TIMEDEFPHASESET), (3, staset)])

  # NOTE that we query arrivals for an additional 2000 seconds beyond the
  # end time
  numidcxarr = export_query(cursor,
                            "select sta, arid, time, deltim, azimuth, "
                            "delaz, slow, delslo, snr, iphase, amp, "
                            "per from idcx%sarrival where time between "
                            "%d and %d order by time"
                            % (options.schemasuffix,
                               start_time, end_time + 2000),
                            tar, "idcx_arrival.csv",
                            [(0, staset), (9, ALLPHASESET)])

  numlebarr = export_query(cursor,
                           "select sta, arid, time, deltim, azimuth, "
                          "delaz, slow, delslo, snr, iphase, amp, "
                           "per from leb%sarrival where time between "
                           "%d and %d order by time"
                           % (options.schemasuffix,
                              start_time, end_time + 2000),
                           tar, "leb_arrival.csv",
                           [(0, staset), (9, ALLPHASESET)])
  
  # create a readme file
  fp = open(TEMPNAME, "wb")
  print >> fp, "Data exported on", datetime.utcnow()
  print >> fp, "Time range", start_dt, "to", end_dt, "(+2000 secs of arrivals)"
  print >> fp, "(%.2f to %.2f and arrivals upto %.2f)" \
        % (start_time, end_time, end_time + 2000)
  print >> fp, "Network", options.network
  print >> fp, "Table static%ssite -- %d sites, %d arrays" \
        % (options.schemasuffix, len(sitedata),
           sum(x[4] for x in sitedata))
  print >> fp, "%d phases %d timedefining" % (len(PHASES),
                                              sum(x[1] for x in PHASES))
  
  print >> fp, "Table leb%sorigin -- %d rows" \
        % (options.schemasuffix, numlebori)
  print >> fp, "Table leb%sassoc -- %d rows" \
        % (options.schemasuffix, numlebass)
  print >> fp, "Table leb%sarrival -- %d rows" \
        % (options.schemasuffix, numlebarr)
  
  print >> fp, "Table sel3%sorigin -- %d rows" \
        % (options.schemasuffix, numsel3ori)
  print >> fp, "Table sel3%sassoc -- %d rows" \
        % (options.schemasuffix, numsel3ass)
  print >> fp, "Table idcx%sarrival -- %d rows" \
        % (options.schemasuffix, numidcxarr)
  
  fp.close()
  tar.add(name = TEMPNAME, arcname = "README.txt")
  
  os.remove(TEMPNAME)
  tar.close()


if __name__ == "__main__":
  main()

