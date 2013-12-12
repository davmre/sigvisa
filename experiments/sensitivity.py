from sigvisa import Sigvisa
import numpy as np

def write_calibration_commands(stations, n):
    f = open('experiments/calibration_cmds.sh', 'w')
    for i in range(n):
        evid = str(int(np.random.rand() * 10000000))

        for sta in stations:
            cmd = "python experiments/logodds_event_sta.py -e %s  -s %s --run_name=everything --run_iteration=2 --phases=P --template_model=param --wiggle_family=dummy\n" % ("fake"+evid, sta)
            f.write(cmd)
    f.close()

def write_event_commands(stations):
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    cursor.execute("select evid from leb_origin where time between (select start_time from dataset where label='test') and (select end_time from dataset where label='test') and mb > 0")
    evids = [5393637,] + list(np.array(cursor.fetchall()).flatten()) # add 2009 dprk event

    f = open('experiments/event_cmds.sh', 'w')

    for evid in evids:
        for sta in stations:
            cmd = "python experiments/logodds_event_sta.py -e %d  -s %s --run_name=everything --run_iteration=2 --phases=P --template_model=param --wiggle_family=dummy\n" % (evid, sta)
            f.write(cmd)

    f.close()


with open('stations.txt', 'r') as f:
    stations = f.read()[:-1].split(',')


poor_uptime_stations = ['SIV', 'PMSA', 'NNA ', 'ATAH', 'RCBR', 'JMIC', 'BBTS', 'SNAA', 'SUR ', 'LSZ ', 'JTS ', 'MBAR', 'OPO ', 'ATD ', 'RES ', 'HNR ', 'ASF ', 'MDT ', 'FRB ', 'SJG ', 'SFJD', 'TSUM', 'QSPA', 'GNI', 'LBTB', 'SADO', 'MLR ', 'LVC ', 'USHA', 'PFO' ]
stations = [sta for sta in stations if sta not in poor_uptime_stations]


write_calibration_commands(stations, 200)
write_event_commands(stations)
