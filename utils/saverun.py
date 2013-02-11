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
# save the results of a run into a file

import sys, tarfile, os, csv

import sigvisa.database.db

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

  if len(sys.argv) not in (2,3):
    print >> sys.stderr, "Usage: python saverun.py [<runid>] <filename>.tar"
    sys.exit(1)

  conn = database.db.connect()
  curs = conn.cursor()

  if len(sys.argv) == 3:
    runid = int(sys.argv[1])
    tarfname = sys.argv[2]
  else:
    curs.execute("select max(runid) from visa_run")
    runid, = curs.fetchone()
    tarfname = sys.argv[1]

  if not tarfname.endswith(".tar"):
    tarfname += ".tar"

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
