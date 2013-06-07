import numpy as np
from math import ceil
import os
import sys
import datetime, pytz, calendar


# sites
SITE_LON_COL, SITE_LAT_COL, SITE_ELEV_COL, SITE_IS_ARRAY, \
    SITE_NUM_COLS = range(4 + 1)


UPTIME_QUANT = 3600                     # 1 hour

MAX_TRAVEL_TIME = 2000.0


def read_uptime(cursor, start_time, end_time, arrival_table="idcx_arrival"):
    cursor.execute("select count(*) from static_siteid")
    numsites, = cursor.fetchone()

    uptime = np.zeros((numsites,
                       int(ceil((MAX_TRAVEL_TIME + end_time - start_time) / UPTIME_QUANT))),
                      bool)

    cursor.execute("select snum, hnum, count(*) from "
                   "(select site.id-1 snum,trunc((arr.time-%d)/%d, 0) hnum "
                   "from %s arr, static_siteid site "
                   "where arr.sta = site.sta and "
                   "arr.time between %d and %d) sitearr group by snum, hnum" %
                   (start_time, UPTIME_QUANT, arrival_table, start_time,
                    end_time + MAX_TRAVEL_TIME))

    for (siteidx, timeidx, cnt) in cursor.fetchall():
        uptime[siteidx, timeidx] = True

    return uptime


def read_all_sites(cursor):

    def to_unixtime(jdate):
        if jdate == -1:
            return sys.float_info.max
        year = jdate / 1000
        day = jdate % 1000
        dt = datetime.datetime(year, 1, 1, tzinfo=pytz.utc) + datetime.timedelta(day - 1)
        return calendar.timegm(dt.timetuple())

    cursor.execute("select s.sta, s.lon, s.lat, s.elev, "
                   "(case s.statype when 'ar' then 1 else 0 end), "
                   "s.ondate, s.offdate, s.refsta "
#                   "(select sid.id from static_siteid sid, static_site s2 "
#                   "where s2.sta=sid.sta and s2.refsta=s.refsta and s2.statype='ar') "
                   "from static_site s order by s.sta")
    results= cursor.fetchall()

    def get_refsta_id(cursor, refsta):
        cursor.execute("select distinct sid.id from static_siteid sid, static_site s "
                       "where s.sta=sid.sta and s.refsta='%s' and s.statype='ar'" % row[7])
        siteid = cursor.fetchall()
        if len(siteid) == 0: # if this really is a non-array station...
            cursor.execute("select sid.id from static_siteid sid "
                           "where sid.sta='%s'" % row[7])
            siteid = cursor.fetchall()
            if len(siteid) == 0:
                return -1
        if len(siteid) > 1:
            print siteid
            raise Exception("multiple siteids for station %s/%s, something weird in the database?" % (row[0], row[7]))
        return siteid[0][0]

    sitedata = []
    i = 0
    refsta_ids = dict()
    for row in results:
        i += 1
        if row[7] not in refsta_ids:
            refsta_ids[row[7]] = get_refsta_id(cursor, row[7])
        if refsta_ids[row[7]] == -1:
            continue
        sitedata.append((row[0], row[1], row[2], row[3], row[4], to_unixtime(row[5]), to_unixtime(row[6]), refsta_ids[row[7]]))
    print i, "rows"
    sitedata = sorted(sitedata, key = lambda x: (x[0], -x[6], -x[7]))

    sitenames = np.array([sd[0] for sd in sitedata])
    sitedata = np.array([sd[1:] for sd in sitedata])

    return sitenames, sitedata

def read_sites_by_name(cursor):
    sites = read_sites(cursor)
    cursor.execute("select sta from static_siteid order by id")
    names = [r[0] for r in cursor.fetchall()]

    # returns stations, name_to_siteid_minus1, siteid_minus1_to_name
    return dict(zip(names, sites)), dict(zip(names, range(len(names)))), names


def read_sites(cursor):
    cursor.execute("select lon, lat, elev, "
                   "(case statype when 'ar' then 1 else 0 end) "
                   "from static_siteid "
                   "order by id")
    return np.array(cursor.fetchall())

def read_phases(cursor):
    cursor.execute("select phase from static_phaseid "
                   "order by id")
    phasenames = np.array(cursor.fetchall())[:, 0]

    cursor.execute("select (case timedef when 'd' then 1 else 0 end) "
                   "from static_phaseid "
                   "order by id")
    phasetimedef = np.array(cursor.fetchall())[:, 0].astype(bool)

    return phasenames, phasetimedef
