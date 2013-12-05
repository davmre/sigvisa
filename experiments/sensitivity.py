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
    cursor.execute("select evid from leb_origin where time between (select start_time from dataset where label='test') and (select end_time from dataset where label='test')")
    evids = np.array(cursor.fetchall()).flatten()

    f = open('experiments/event_cmds.sh', 'w')

    for evid in evids:
        for sta in stations:
            cmd = "python experiments/logodds_event_sta.py -e %d  -s %s --run_name=everything --run_iteration=2 --phases=P --template_model=param --wiggle_family=dummy\n" % (evid, sta)
            f.write(cmd)

    f.close()


with open('stations.txt', 'r') as f:
    stations = f.read()[:-1].split(',')

write_calibration_commands(stations, 50)
#write_event_commands(stations)
