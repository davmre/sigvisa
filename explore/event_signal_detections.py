import numpy as np
import sys
import os
import traceback
import pickle
import copy

from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa import Sigvisa
from sigvisa.signals.common import Waveform
from sigvisa.signals.io import load_event_station_chan,fetch_waveform
from sigvisa.source.event import get_event
from sigvisa.models.ttime import tt_predict
from sigvisa.infer.optimize.optim_utils import construct_optim_params
from sigvisa.plotting.plot import plot_with_fit

def main():

    evid = int(sys.argv[1])
    ev = get_event(evid=evid)

    with open('stations_lines.txt', 'r') as f:
        stalist = [s.strip() for s in f.readlines()]

    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sets = []
    for i in range(16):
        sets.append(set())

    f = open('detections.txt', 'w')

    leb_stas = []
    for station in stalist:

        if s.earthmodel.site_info(station, 0)[3] == 1:
            cursor.execute("select refsta from static_site where sta='%s'" % station)
            sta = cursor.fetchone()[0]
        else:
            sta = station

        try:
            atime = tt_predict(ev, sta, 'P') + ev.time
            stime = atime - 50
            etime = atime + 300


            """
            wave = fetch_waveform(station=sta, stime=stime, etime=etime, chan="auto", cursor=cursor).filter("%s;env" % "freq_2.0_3.0").filter('hz_5.0')
            print "loaded wave at", sta


            sg = SigvisaGraph(template_model_type="gplocal+lld+dist5", template_shape="paired_exp",
                          wiggle_model_type="dummy", wiggle_family="dummy",
                          phases=["P",], nm_type = "ar", assume_envelopes=True,
                          wiggle_len_s = 60.0, arrays_joint=False, runid=4, iteration=1, base_srate=5.0,
                          gpmodel_build_trees=False)


            wn = sg.add_wave(wave)
            lp1 = sg.current_log_p(verbose=True)

            ev = get_event(evid=evid)
            ev_node = sg.add_event(ev)
            ev_prior_lp = ev_node.log_p()
            lp11 = lp1 + ev_prior_lp

            sg.parent_predict_all()
            sg.optimize_templates(construct_optim_params("'disp': True, 'method': 'bfgs'"))
            lp2 = sg.current_log_p()
            """
            sql_query = "select l.time, l.snr from leb_arrival l, leb_origin lebo, leb_assoc leba where l.sta='%s' and l.time between %f and %f and l.arid=leba.arid and leba.orid=lebo.orid and lebo.evid=%d" % (station, atime-50, atime+50, evid)
            cursor.execute(sql_query)
            leb_arrivals = cursor.fetchall()
            if len(leb_arrivals) > 0:
                print "appending", station
                leb_stas.append(station)
            continue

            sql_query = "select l.time, l.snr from sel3_arrival l, sel3_origin lebo, sel3_assoc leba where l.sta='%s' and l.time between %f and %f and l.arid=leba.arid and leba.orid=lebo.orid and lebo.evid=%d" % (station, atime-50, atime+50, evid)
            cursor.execute(sql_query)
            sel3_arrivals = cursor.fetchall()

            sql_query = "select ia.time, ia.snr from idcx_arrival ia where ia.sta='%s' and ia.time between %f and %f" % (station, atime-50, atime+50)
            cursor.execute(sql_query)
            idc_arrivals = cursor.fetchall()

            """
            v, tg = wn.get_template_params_for_arrival(eid=0, phase='P')
            snr = np.exp(v['coda_height']) / wn.nm.c
            """

            code = 0
            if len(leb_arrivals) > 0:
                code += 1
            if len(idc_arrivals) > 0:
                code += 2
            if len(sel3_arrivals) > 0:
                code += 4

            f.write('\n%s\n------\n' % station)
            """f.write("SIGVISA %.1f %.1f %.1f %.1f\n" % (v['arrival_time'], snr, lp11, lp2))
            if lp2 > lp11:
                f.write( "\t detected at %s!\n" % (sta))
                code += 8
            else:
                f.write("\n")"""
            f.write( "LEB %s\n" % leb_arrivals)
            f.write( "IDC %s\n" % idc_arrivals)
            f.write( "SEL3 %s\n" % sel3_arrivals)

            sets[code].add(station)

            #plot_with_fit('logs/dprk_detection/%s.png' % station, wn)

            f.flush()
            f1 = open('detection_sets.txt', 'w')
            for i in range(16):
                if (i % 2) == 1:
                    f1.write(" LEB")
                if (i % 4) >= 2:
                    f1.write(" IDC")
                if (i % 8) >= 4:
                    f1.write(" SEL3")
                if (i % 16) >= 8:
                    f1.write(" SIGVISA")
                f1.write(': ')
                for set_sta in sets[i]:
                    f1.write(set_sta + ',')
                f1.write('\n')
            f1.flush()
            f1.close()
        except Exception as e:
            f.write('error: %s\n' % e)
            print "error at %s: %s" % (sta, e)
            continue
    print leb_stas
    f.close()



if __name__ == "__main__":
    try:
        #sample_template()
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        import pdb
        pdb.post_mortem(tb)
