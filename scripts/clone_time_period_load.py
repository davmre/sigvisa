from sigvisa.database import db
from sigvisa import Sigvisa
import numpy as np
import sigvisa.utils.geog
import csv
from optparse import OptionParser
import os
import sys


from sigvisa.utils.fileutils import mkdir_p

def ensure_path_exists(fname):
    dirname = os.path.dirname(fname)
    mkdir_p(dirname)

def load_table(cursor, table_name, fname):
    sql_query = "load data local infile '%s' into table %s fields terminated by ','" % (os.path.abspath(fname), table_name)
    print sql_query
    cursor.execute(sql_query)

def fix_wfdisc(fname, dest, src="/home/dmoore/ctbt_data/seismic"):
    with open(fname, 'r') as f:
        lines = f.readlines()
    fixed_lines = [line.replace(src,dest) for line in lines]
    with open(fname, 'w') as f:
        for line in fixed_lines:
            f.write(line)
    print "fixed", fname



def main():

    wfdisc_fname = "idcx_wfdisc.csv"

    basedir = sys.argv[1]

    dest_basedir = "/home/dmoore/ctbt_data"
    wfdest = os.path.join(dest_basedir, basedir)
    wfsrc= os.path.join(basedir, "seismic")

    s = Sigvisa()
    cursor = db.connect().cursor()

    """
    for sta in os.listdir(basedir):
        if sta=="seismic": continue
        print sta
        stadir = os.path.join(basedir, sta)
        files = os.listdir(stadir)

        if "idcx_wfdisc.csv" in files:
            fpath = os.path.join(stadir, "idcx_wfdisc.csv")
            fix_wfdisc(fpath, wfdest, wfsrc)
            load_table(cursor, 'idcx_wfdisc', fpath)

        if "idcx_arrival.csv" not in files:
            print "no arrivals found, skipping"
            continue
        load_table(cursor, 'idcx_arrival', os.path.join(stadir, 'idcx_arrival.csv'))
        load_table(cursor, 'leb_arrival', os.path.join(stadir, 'leb_arrival.csv'))
        load_table(cursor, 'leb_origin', os.path.join(stadir, 'leb_origin.csv'))
        load_table(cursor, 'leb_assoc', os.path.join(stadir, 'leb_assoc.csv'))
    """
    mkdir_p(wfdest)
    cmd = "cp -R %s/* %s" % (wfsrc , wfdest)
    print cmd
    os.system(cmd)

    cursor.close()


if __name__ == "__main__":
    main()
