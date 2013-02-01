import database.db
from database.dataset import *
import utils.geog
import sys
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-e", "--evid", dest="evid", default=None, type="int", help="event ID")
parser.add_option("--ss_only", dest="ss_only", default=False, action="store_true", help="list only three-component stations, no arrays")
(options, args) = parser.parse_args()

cursor = db.connect().cursor()

evid = options.evid

cursor = database.db.connect().cursor()
phases = read_phases(cursor)[0]

sites, name2siteid, siteid2name = read_sites_by_name(cursor)

ev = read_event(cursor, evid)

ss_cond = "and site.statype='ss'" if options.ss_only else ""

sql_query = "select site.id-1, iarr.arid, iarr.time, iarr.deltim, iarr.azimuth, iarr.delaz, iarr.slow, iarr.delslo, iarr.snr, ph.id-1, iarr.amp, iarr.per from leb_arrival iarr, leb_assoc iass, leb_origin ior, static_siteid site, static_phaseid ph where iarr.snr > 0 and iarr.sta=site.sta and iarr.iphase=ph.phase and ascii(iarr.iphase) = ascii(ph.phase) and iarr.arid=iass.arid and iass.orid=ior.orid and ior.evid=%d %s order by iarr.snr desc" %  (evid, ss_cond)

cursor.execute(sql_query)
dets = cursor.fetchall()

print "sta\tsiteid\tarr\tphase\tsnr\tdist\t|\ttime\t\tamp\tazi\tslo"
for det in dets:
    sta = siteid2name[det[DET_SITE_COL]]
    site_ll = sites[sta][0:2]
    site_type = 'ar' if sites[sta][3]==1 else 'ss'
    ev_ll = ev[0:2]
    dist = utils.geog.dist_km(site_ll, ev_ll)
    phase = phases[det[DET_PHASE_COL]]
    print "%s\t%d\t%s\t%s\t%.2f\t%.1f\t|\t%.2f\t%.2f\t%.2F\t%.2F" % (sta, det[DET_SITE_COL]+1, site_type, phase, det[DET_SNR_COL], dist,  det[DET_TIME_COL], det[DET_AMP_COL], det[DET_AZI_COL], det[DET_SLO_COL])
