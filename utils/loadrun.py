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
# load the results of a run from a file

import sys, tarfile, os, csv

import database.db

def importtable(curs, tar, fname, tabname, inp_runid=None):
  fobj = tar.extractfile(fname)
  inp = csv.reader(fobj)
  
  colnames = inp.next()

  for runid_pos in xrange(len(colnames)):
    if colnames[runid_pos].lower() == "runid":
      break
  else:
    print >> sys.stderr, "No runid in table"
    sys.exit(1)

  if inp_runid is None:
    colnames.pop(runid_pos)

  query = "insert into %s (%s) values (%s)" % (tabname, ",".join(colnames),
                                           ",".join(["%s" for _ in colnames]))
  
  for line in inp:
    if inp_runid is None:
      line.pop(runid_pos)
    else:
      line[runid_pos] = str(inp_runid)

    for i in range(len(line)):
      if line[i] == '':
        line[i] = 'null'
      else:
        line[i] = "'" + line[i] + "'"
    
    curs.execute(query % tuple(line))
    
  
  if inp_runid is None:
    curs.execute("select max(runid) from visa_run")
    runid, = curs.fetchone()
    
    return runid

def main():
  
  if len(sys.argv) != 2:
    print >> sys.stderr, "Usage: python loadrun.py <filename>.tar"
    sys.exit(1)
  
  tarfname = sys.argv[1]
  if not tarfname.endswith(".tar"):
    tarfname += ".tar"
  
  conn = database.db.connect()
  curs = conn.cursor()
  
  # grab the next runid
  tar = tarfile.open(name = tarfname, mode='r')
  
  runid = importtable(curs, tar, "visa_run.csv", "visa_run")

  print "RUNID", runid

  importtable(curs, tar, "visa_origin.csv", "visa_origin", runid)

  importtable(curs, tar, "visa_assoc.csv", "visa_assoc", runid)
  
  tar.close()
  
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

