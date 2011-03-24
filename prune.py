# prunes events which are within a space-time ball of better events
import os, sys
import numpy as np
from optparse import OptionParser

from database.dataset import *
from analyze import suppress_duplicates
import database.db

def main():
  parser = OptionParser()
  parser.add_option("-i", "--runid", dest="runid", default=None,
                    type="int",
                    help = "the run-identifier to prune (last runid)")

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
  
