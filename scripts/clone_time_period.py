from sigvisa.database import db
from sigvisa import Sigvisa
import numpy as np
import sigvisa.utils.geog
import csv
from optparse import OptionParser
import os

cursor = db.connect().cursor()
from sigvisa.utils.fileutils import mkdir_p

def ensure_path_exists(fname):
    dirname = os.path.dirname(fname)
    mkdir_p(dirname)


def dump_table(rows, fname):
    ensure_path_exists(fname)

    csvWriter = csv.writer(open(fname, 'wb'), delimiter=',',
                           quotechar="'", quoting=csv.QUOTE_MINIMAL)
    nrows = 0
    print "writing to %s..." % (fname,)
    for r in rows:
        csvWriter.writerow(r)
        nrows = nrows + 1
        if nrows % 10000 == 0:
            print "... wrote %d rows" % nrows
    print "... wrote %d rows" % nrows
    print "done."

def strip_fname(dirname, base_fname, basedir):
    assert( dirname.startswith("/archive/ops"))
    return os.path.join(basedir, dirname[13:], base_fname)

def read_data(fname, sidx, eidx):
    try:
        datafile = open(fname, "rb")
    except IOError, e:
        print "cannot open file ", fname
        # the file could be compressed try .gz extension
        datafile = gzip.open(fname + ".gz")

    # seek to the desired offset
    datafile.seek(sidx)
    # and read the number of bytes required
    bytes = datafile.read(eidx-sidx)
    datafile.close()
    return bytes

def write_data(fname, d):
    ensure_path_exists(fname)
    datafile = open(fname, "wb")
    datafile.write(d)
    datafile.close()

def dump_waveforms(cursor, sta, start_time, end_time, basedir=""):
    s = Sigvisa()
    sta = s.get_default_sta(sta)
    print "getting waveforms for", sta

    cond = "sta='%s' and ((time > %f and time < %f) or (endtime > %f and endtime < %f) or (time < %f and endtime > %f))" % (sta, start_time, end_time, start_time, end_time, start_time, end_time)
    sql_query = "select dir,dfile,foff,nsamp,datatype from idcx_wfdisc where " + cond
    cursor.execute(sql_query)
    file_ranges = dict()
    for (ddir, dfile, foff, nsamp, dtype) in cursor:
        full_fname = os.path.join(ddir, dfile)
        assert(dtype == "s3" or dtype == "s4")
        bytes_per_sample = int(dtype[-1])
        nbytes = nsamp * bytes_per_sample
        foff_end = foff+nbytes
        if full_fname not in file_ranges:
            sidx, eidx = (foff, foff_end)
        else:
            sidx, eidx = file_ranges[full_fname]
            sidx = min(sidx, foff)
            eidx = max(eidx, foff_end)
        file_ranges[full_fname] = (sidx, eidx)

    fname_map = dict()
    for fname, (sidx, eidx) in file_ranges.items():
        dirname, base_fname = os.path.split(fname)
        d = read_data(fname, sidx, eidx)
        print "read data from", fname, sidx, eidx
        new_fname = strip_fname(dirname, base_fname, basedir)
        write_data(new_fname, d)
        print "wrote to", new_fname
        fname_map[fname] = (new_fname, sidx)

    sql_query = "select * from idcx_wfdisc where " + cond
    cursor.execute(sql_query)
    rows = []
    for row in cursor:
        ddir, dfname, foff = row[15], row[16], row[17]
        full_fname = os.path.join(ddir, dfname)
        new_fname, offset = fname_map[full_fname]
        new_dir, new_dfname = os.path.split(new_fname)
        lrow = list(row)
        lrow[15], lrow[16], lrow[17] = new_dir, new_dfname, foff-offset

        rows.append(lrow)

    fname = os.path.join(basedir, sta, "idcx_wfdisc.csv")
    dump_table(rows, fname)


def dump_arrivals(cursor, sta, start_time, end_time, basedir=""):
    sql_query = "select * from idcx_arrival where sta='%s' and time between %f and %f" % (sta, start_time, end_time)
    fname = os.path.join(basedir, sta, "idcx_arrival.csv")
    cursor.execute(sql_query)
    dump_table(cursor, fname)

def dump_leb(cursor, sta, start_time, end_time, basedir=""):
    sql_query = "select * from leb_arrival where sta='%s' and time between %f and %f" % (sta, start_time, end_time)
    arrival_fname = os.path.join(basedir, sta, "leb_arrival.csv")
    print sql_query
    cursor.execute(sql_query)
    dump_table(cursor, arrival_fname)

    sql_query = "select arid from leb_arrival where sta='%s' and time between %f and %f" % (sta, start_time, end_time)
    cursor.execute(sql_query)
    arids = cursor.fetchall()
    assoc_rows = []
    origin_rows = []
    for arid in arids:
        sql_query = "select * from leb_assoc where arid=%d" % arid
        cursor.execute(sql_query)
        orids = []

        for row in cursor:
            orid = row[1]
            orids.append(orid)
            assoc_rows.append(row)

        for orid in orids:
            sql_query = "select * from leb_origin where orid=%d" % orid
            cursor.execute(sql_query)
            for row in cursor:
                origin_rows.append(row)

    assoc_fname = os.path.join(basedir, sta, "leb_assoc.csv")
    dump_table(assoc_rows, assoc_fname)
    origin_fname = os.path.join(basedir, sta, "leb_origin.csv")
    dump_table(origin_rows, origin_fname)


def main():

    parser = OptionParser()

    parser.add_option("--stas", dest="stas", default="", type="str")
    parser.add_option("--start_time", dest="start_time", default=None, type="float")
    parser.add_option("--end_time", dest="end_time", default=None, type="float")
    (options, args) = parser.parse_args()

    stas = options.stas.split(',')

    cursor = db.connect().cursor()
    basedir = "vdec_dump_%.1f_%.1f" % (options.start_time, options.end_time)
    try:
        for sta in stas:
            dump_waveforms(cursor, sta, options.start_time, options.end_time, basedir)
            dump_leb(cursor, sta, options.start_time, options.end_time, basedir)
            dump_arrivals(cursor, sta, options.start_time, options.end_time, basedir)
    finally:
        cursor.close()

if __name__ == "__main__":
    main()
