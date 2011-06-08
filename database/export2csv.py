# converts an oracle table into a csv file (requires python and cx_Oracle)
# inspired by: http://stackoverflow.com/questions/65447/getting-data-from-an-oracle-database-as-a-csv-file-or-any-other-custom-text-form
#
# example:
# python export2csv.py oracle user/pass tablename filename.csv
#   OR
# python export2csv.py mysql user:ctbt,db=ctbt3mos tablename filename.csv

from optparse import OptionParser
import sys, csv

def main():
  parser = OptionParser(usage = "Usage: %prog [options] db-type connect-args"
                        + "db-table csv-filename")
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
    connargs = dict(pair.split(":") for pair in args[1].split(","))
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
  
