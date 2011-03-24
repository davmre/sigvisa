# converts an oracle table into a csv file (requires python and cx_Oracle)
# inspired by: http://stackoverflow.com/questions/65447/getting-data-from-an-oracle-database-as-a-csv-file-or-any-other-custom-text-form
#
# example:
# python oracle2csv.py user/pass tablename filename.csv

from optparse import OptionParser
import sys
import cx_Oracle, csv

def main():
  parser = OptionParser(usage = "Usage: %prog [options] connect-string "
                        + "db-table csv-filename")
  parser.add_option("-o", "--orderby", dest="orderby",
                    help="order the output by these columns (e.g. a,b)",
                    metavar = "COLLIST")
  parser.add_option("-w", "--where", dest="where",
                    help="where clause (e.g. a>5)",
                    metavar = "CLAUSE")
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.print_help()
    sys.exit(1)
  
  orcl = cx_Oracle.connect(args[0])
  convert_table(orcl, args[1], args[2], options.where, options.orderby)

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
  
