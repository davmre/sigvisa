from sigvisa import Sigvisa
from sigvisa.utils.geog import dist_km
import numpy as np

stas = ['AAK', 'AKTO', 'BVAR', 'CMAR', 'KURK', 'MKAR', 'SONM', 'ZALV']

def get_evids_in_region(sta, left, right, bottom, top):
    sql_query = "select lebo.evid, lebo.lon, lebo.lat, lebo.depth, lebo.time, lebo.mb from leb_origin lebo, leb_assoc leba, leb_arrival l where l.arid=leba.arid and leba.orid=lebo.orid and l.sta='%s' and lebo.mb > 4.5 and leba.phase='Pn' and lebo.lon between %f and %f and lebo.lat between %f and %f" % (sta, left, right, bottom, top)

    s = Sigvisa()
    cursor = s.dbconn.cursor()
    cursor.execute(sql_query)
    r = cursor.fetchall()
    cursor.close()

    slon, slat = s.earthmodel.site_info(sta, 0)[0:2]
    def filter_dist(lon, lat):
        d = dist_km((lon, lat), (slon, slat))
        return d > 300 and d < 2000

    print "got %d results, filtering..." % len(r)
    r_filtered = [rr for rr in r if filter_dist(rr[1], rr[2])]

    print "got %d results after filtering." % len(r_filtered)

    return r_filtered

def main():
    r = get_evids_in_region('MKAR', 60.0, 100.0, 35.0, 50.0)
    np.save('r.npy', r)

if __name__ == "__main__":
    main()
