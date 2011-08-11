# load the results of a run from a file

import sys, tarfile, os, csv

import database.db

def importtable(curs, tar, fname, tabname, inp_runid=None):
  fobj = tar.extractfile(fname)
  inp = csv.reader(fobj)
  
  colnames = inp.next()

  for runid_pos in xrange(len(colnames)):
    if colnames[runid_pos] == "runid":
      break
  else:
    print >> sys.stderr, "No runid in table"
    sys.exit(1)

  if inp_runid is None:
    colnames.pop(runid_pos)

  query = "insert into %s (%s) values (%s)" % (tabname, ",".join(colnames),
                                           ",".join(["'%s'" for _ in colnames]))
  
  for line in inp:
    if inp_runid is None:
      line.pop(runid_pos)
    else:
      line[runid_pos] = str(inp_runid)

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
  main()
