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
# converts an oracle table into a csv file (requires python and cx_Oracle)
# inspired by: http://stackoverflow.com/questions/65447/getting-data-from-an-oracle-database-as-a-csv-file-or-any-other-custom-text-form
#
# example:
# python export2csv.py oracle user/pass tablename filename.csv
#   OR
# python export2csv.py mysql user=ctbt,db=ctbt3mos tablename filename.csv

from optparse import OptionParser
import sys, csv

def main():
  parser = OptionParser(usage = "Usage: %prog [options] db-type connect-args"
                        + " db-table csv-filename")
  parser.add_option("-o", "--orderby", dest="orderby",
                    help="order the output by these columns (e.g. a,b)",
                    metavar = "COLLIST")
  parser.add_option("-w", "--where", dest="where",
                    help="where clause (e.g. a>5)",
                    metavar = "CLAUSE")
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.print_help()
    sys.exit(1)

  if args[0] == "oracle":
    import cx_Oracle
    db = cx_Oracle.connect(args[1])
  elif args[0] == "mysql":
    import MySQLdb
    connargs = dict(pair.split("=") for pair in args[1].split(","))
    db = MySQLdb.connect(**connargs)
  
  convert_table(db, args[2], args[3], options.where, options.orderby)

def convert_table(orcl, tablename, filename, where, orderby):
  
  curs = orcl.cursor()

  sql = "select * from %s" % tablename

  if where is not None:
    sql += " where %s" % where
  
  if orderby is not None:
    sql += " order by %s" % orderby
  
  curs.execute(sql)
  
  output = csv.writer(open(filename, 'wb'))
  
  output.writerow([col[0] for col in curs.description])
  
  for row_data in curs:        
    output.writerow(row_data)

if __name__ == "__main__":
  main()
  
