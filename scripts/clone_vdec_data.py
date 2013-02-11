from sigvisa.database import db
import numpy as np
import sigvisa.utils.geog
import csv
from optparse import OptionParser
import os

cursor = db.connect().cursor()

def dump_table(cursor, table_name):
    sql_query = "select * from %s" % table_name
    cursor.execute(sql_query)

    fname = '%s.csv' % table_name
    csvWriter = csv.writer(open(fname, 'wb'), delimiter=',',
                           quotechar="'", quoting=csv.QUOTE_MINIMAL)

    nrows = 0
    print "writing table %s to %s..." % (table_name, fname)
    for r in cursor:
        csvWriter.writerow(r)
        nrows= nrows + 1
        if nrows % 10000 == 0:
            print "... wrote %d rows" % nrows
    print "... wrote %d rows" % nrows
    print "done."


def clear_table(cursor, table_name):
    sql_query = "delete from %s" % table_name
    print sql_query
    cursor.execute(sql_query)

def load_table(cursor, table_name, fname):
    sql_query = "load data infile '%s' into table %s fields terminated by ','" % (os.path.abspath(fname), table_name)
    print sql_query
    cursor.execute(sql_query)

def main():

    parser = OptionParser()

    parser.add_option("--dump", dest="dump", default=False, action="store_true")
    parser.add_option("--import", dest="load", default=False, action="store_true")
    parser.add_option("--preserve", dest="preserve", default=False, action="store_true")
    (options, args) = parser.parse_args()

    if (options.dump and options.load)  or (not options.dump and not options.load):
        raise Exception("must specify exactly one of --dump or --import")

    cursor = db.connect().cursor()

    if options.dump:
        if len(args) == 0:
            args = ["sigvisa_coda_fits", "sigvise_wiggle_wfdisc"]
        for table in args:
            dump_table(cursor, table)
    elif options.load:
        if len(args) == 0:
            raise Exception("must specify list of .csv files to import...")
        for fname in args:
            a,b = os.path.splitext(fname)
            if b != ".csv":
                raise Exception("filename must be in format <db_table_name>.csv (got %s)" % fname)
            tname = os.path.split(a)[-1]

            if not options.preserve:
                try:
                    clear_table(cursor, tname)
                except Exception as e:
                    print "could not clear table %s: exception" % tname, e
            try:
                load_table(cursor, tname, fname)
            except Exception as e:
                print "could not load into table %s: exception" % tname, e

if __name__ == "__main__":
    main()
