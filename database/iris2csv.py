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

import sys, calendar

def main():
  if len(sys.argv) != 2:
    print "Error: Usage python iris2csv.py <IRIS-weed bulletin>"
    sys.exit(1)

  in_fname = sys.argv[1]
  out_fname = in_fname+".csv"

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
    import pdb, traceback, sys
    traceback.print_exc(file=sys.stdout)
    pdb.post_mortem(sys.exc_traceback)
    raise
  
