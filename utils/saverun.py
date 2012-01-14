# save the results of a run into a file

import sys, tarfile, os, csv

import database.db

TEMPNAME = "tempsave.csv"

def export(curs, tablename, runid, filename):
  curs.execute("select * from %s where runid=%d" % (tablename, runid))
  rows = curs.fetchall()

  output = csv.writer(open(filename, "wb"))
  
  output.writerow([col[0] for col in curs.description])
  
  for row in rows:
    output.writerow(row)

  del output                            # close the file

  return len(rows)

def main():
  
  if len(sys.argv) != 3:
    print >> sys.stderr, "Usage: python saverun.py <runid> <filename>.tar"
    sys.exit(1)
  
  runid = int(sys.argv[1])
  tarfname = sys.argv[2]
  if not tarfname.endswith(".tar"):
    tarfname += ".tar"
  
  conn = database.db.connect()
  curs = conn.cursor()

  if export(curs, "visa_run", runid, TEMPNAME) != 1:
    print >> sys.stderr, "Runid %d not found" % runid
    sys.exit(1)

  tar = tarfile.open(name = tarfname, mode='w')
  
  tar.add(name = TEMPNAME, arcname="visa_run.csv")
  
  export(curs, "visa_origin", runid, "tempsave.csv")

  tar.add(name = TEMPNAME, arcname="visa_origin.csv")

  export(curs, "visa_assoc", runid, "tempsave.csv")

  tar.add(name = TEMPNAME, arcname="visa_assoc.csv")

  tar.close()

  os.remove(TEMPNAME)
  
  return

  
if __name__ == "__main__":
  main()
