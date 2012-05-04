import os, errno, sys, time, traceback
import numpy as np, scipy

from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from optparse import OptionParser

import plot
import learn, sigvisa_util
import signals.SignalPrior
from utils.waveform import *
import utils.geog
import obspy.signal.util


import utils.nonparametric_regression as nr
from priors.coda_decay.coda_decay_common import *
from priors.coda_decay.train_coda_models import CodaModel


def plot_channels_with_pred(pp, vert_trace, vert_params, phaseids, horiz_trace, horiz_params, title = None):
    fig = plt.figure(figsize = (8, 8))

    bhz_axes = plt.subplot(2, 1, 1)

    if title is not None:
        plt.title(title, fontsize=12)

    plot_envelopes_with_pred(bhz_axes, vert_trace, phaseids, vert_params)
    horiz_axes = plt.subplot(2, 1, 2, sharex=bhz_axes, sharey = bhz_axes)
    plot_envelopes_with_pred(horiz_axes, horiz_trace, phaseids, horiz_params)

    pp.savefig()
    plt.close(fig)

def plot_envelopes_with_pred(axes, trace, phaseids, params):
    srate = trace.stats['sampling_rate']

    synth_trace = imitate_envelope(trace, phaseids, params)

    traces = [trace,synth_trace]
    formats = ["k-","g-"]
    linewidths = [5,5]

    plot.plot_traces_subplot(axes, traces, formats = formats, linewidths = linewidths)

def plot_channels(pp, vert_trace, vert_noise_floor, vert_fits, vert_formats, horiz_trace, horiz_noise_floor, horiz_fits, horiz_formats, all_det_times = None, all_det_labels = None, title = None):
    fig = plt.figure(figsize = (8, 8))

    bhz_axes = plt.subplot(2, 1, 1)

    if title is not None:
        plt.title(title, fontsize=12)

    plot_envelopes(bhz_axes, vert_trace, vert_noise_floor, vert_fits, vert_formats, all_det_times, all_det_labels)
    horiz_axes = plt.subplot(2, 1, 2, sharex=bhz_axes, sharey = bhz_axes)
    plot_envelopes(horiz_axes, horiz_trace, horiz_noise_floor, horiz_fits, horiz_formats, all_det_times, all_det_labels)

    pp.savefig()
    plt.close(fig)

def plot_envelopes(axes, trace, noise_floor, fits, formats, all_det_times = None, all_det_labels = None):
    srate = trace.stats['sampling_rate']

#    siteid = int(arrival[AR_SITEID_COL])
#    phaseid = int(arrival[AR_PHASEID_COL])

    traces = [trace,]
    formats = ["k-",] + formats
    linewidths = [5,]

    for fit in fits:
        if fit is not None and fit[FIT_CODA_LENGTH] > 0:
            stats = trace.stats.copy()
            stats['starttime_unix'] += fit[FIT_CODA_START_OFFSET]
            fit_trace = Trace(gen_logenvelope(fit[FIT_CODA_LENGTH], srate, fit[FIT_HEIGHT], 0, fit[FIT_B]), stats)
            fit_trace.stats.npts = len(fit_trace.data)
            traces.append(fit_trace)
#            formats.append("r-")
            linewidths.append(5)

#    pred_stats = trace.stats.copy()
#    pred_stats['starttime_unix'] = arrival[8] + netmodel.mean_travel_time(arrival[2], arrival[3], arrival[9], siteid-1, phaseid-1)
#    pred_para_b = unfair_para_predict(shape_params, arrival, band, distance)[0]
#    pred_nonpara_b = fair_nonpara_predict(arrival, lb, 30)

#    pred_trace_para = Trace(gen_logenvelope(coda_length, srate, gamma, 0, pred_para_b), pred_stats)
#    pred_trace_para.stats.npts = len(pred_trace_para.data)

#    if pred_nonpara_b is not None:
#        pred_trace_nonpara = Trace(gen_logenvelope(coda_length, srate, gamma, 0, pred_nonpara_b), pred_stats.copy())
#    else:
#        pred_trace_nonpara = Trace(np.array(()), pred_stats.copy())
#    pred_trace_nonpara.stats.npts = len(pred_trace_nonpara.data)


    plot.plot_traces_subplot(axes, traces, all_det_times=all_det_times, all_det_labels=all_det_labels, formats = formats, linewidths = linewidths)
#    plot.plot_traces([trace, fake_trace, pred_trace_para, pred_trace_nonpara], title=title + " pb %f npb %f" % (pred_para_b, pred_nonpara_b), all_det_times=all_det_times, all_det_labels=all_det_labels, formats = ["k-", "r-", "b:", "g:"], linewidths = [5,5,1,1])
    maxtrc, mintrc = float(max(trace.data)), float(min(trace.data))
#    plt.bar(left = trace.stats['starttime_unix'] + phase_start_time - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)
#    plt.bar(left = trace.stats['starttime_unix'] + (peak_offset_time + coda_length) - .2, height = maxtrc-mintrc, width=.25, bottom=mintrc, color="blue", linewidth=1, alpha=.5)

    xvals = [trace.stats.starttime_unix, trace.stats.starttime_unix + trace.stats.npts/srate]
    axes.plot(xvals, [noise_floor, noise_floor], "g-")


def plot_scatter(lp, ls, lsp, base_coda_dir, band):
    try:

        pdf_dir = get_dir(os.path.join(base_coda_dir, band[19:]))
        pp = PdfPages(os.path.join(pdf_dir, "plots.pdf"))
        print "opening pp in ", pdf_dir

        if lp is not None and len(lp.shape) == 2:
            plt.figure()
            plt.title("P codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            plt.plot(lp[:, 0], lp[:, 3], 'ro')
            pp.savefig()

        if ls is not None and len(ls.shape) == 2:
            plt.figure()
            plt.title("S codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            plt.plot(ls[:, 0], ls[:, 3], 'ro')
            pp.savefig()

        if lp is not None and len(lp.shape) == 2:
            plt.figure()
            plt.title("tele P codas/depth")
            plt.xlabel("depth (km)")
            plt.ylabel("b")
            tele_i = (lp[:, 0] > 1000)
            plt.plot(lp[tele_i, 2], lp[tele_i, 3], 'ro')
            pp.savefig()

        if ls is not None and len(ls.shape) == 2:
            plt.figure()
            plt.title("tele S codas/depth")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            tele_i = (ls[:, 0] > 1000)
            plt.plot(ls[tele_i, 2], ls[tele_i, 3], 'ro')
            pp.savefig()


        if lp is not None and len(lp.shape) == 2:
            plt.figure()
            plt.title("shallow (0-30km) P codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            shallow_i = (lp[:, 2] < 30)
            plt.plot(lp[shallow_i, 0], lp[shallow_i, 3], 'ro')
            pp.savefig()

        if ls is not None and len(ls.shape) == 2:
            plt.figure()
            plt.title("shallow (0-30km) S codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            shallow_i = (ls[:, 2] < 30)
            plt.plot(ls[shallow_i, 0], ls[shallow_i, 3], 'ro')
            pp.savefig()

        if lp is not None and len(lp.shape) == 2:
            plt.figure()
            plt.title("deep (100+ km) P codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            deep_i = (lp[:, 2] > 100)
            plt.plot(lp[deep_i, 0], lp[deep_i, 3], 'ro')
            pp.savefig()

        if ls is not None and len(ls.shape) == 2:
            plt.figure()
            plt.title("deep (100+km) S codas/distance")
            plt.xlabel("distance (km)")
            plt.ylabel("b")
            deep_i = (ls[:, 2] > 100)
            plt.plot(ls[deep_i, 0], ls[deep_i, 3], 'ro')
            pp.savefig()

        if lp is not None and len(lp.shape) == 2:
            plt.figure()
            plt.title("P codas / azimuth")
            plt.xlabel("azimuth (deg)")
            plt.ylabel("b")
            plt.xlim([0, 360])
            plt.plot(lp[:, 1], lp[:, 3], 'ro')
            pp.savefig()

        if ls is not None and len(ls.shape) == 2:
            plt.figure()
            plt.title("S codas / azimuth")
            plt.xlabel("azimuth (deg)")
            plt.ylabel("b")
            plt.xlim([0, 360])
            plt.plot(ls[:, 1], ls[:, 3], 'ro')
            pp.savefig()

        if lsp is not None and len(lsp.shape) == 2:
            plt.figure()
            plt.title("P vs S")
            plt.xlabel("b (P coda)")
            plt.ylabel("b (S coda)")
            plt.plot(lsp[:, 0], lsp[:, 1], 'ro')
            pp.savefig()

    except:
        print "error plotting learned params for ", band
        print traceback.format_exc()
    finally:
        pp.close()


def generate_scatter_plots(all_data, bands, base_coda_dir):
    for (band_idx,band) in enumerate(bands):
        short_band = band[19:]

        band_data = extract_band(all_data, band_idx)

#        print "band data", band_data

        lp = None
        ls = None
        lsp = None
        for row in band_data:
            fit_p_vert = fit_from_row(row, P=True, vert=True)
            fit_s_horiz = fit_from_row(row, P=False, vert=False)

            accept_p_vert = accept_fit(fit_p_vert, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)
            if accept_p_vert:
                r = np.array((row[DISTANCE_COL], row[AZI_COL], row[DEPTH_COL], fit_p_vert[FIT_B]))
                if lp == None:
                    lp = r
                else:
                    lp = np.vstack([lp, r])

                print "%s: accepted p for %d" % (short_band, row[EVID_COL])
            else:
                print "%s: rejected p for %d" % (short_band, row[EVID_COL])

            accept_s_horiz = accept_fit(fit_s_horiz, min_coda_length=min_s_coda_length, max_avg_cost = avg_cost_bound)
            if accept_s_horiz:
                r = np.array((row[DISTANCE_COL], row[AZI_COL], row[DEPTH_COL], fit_s_horiz[FIT_B]))
                if ls == None:
                    ls = r
                else:
                    ls = np.vstack([ls, r])
#                print "%s: accepted s for %d" % (short_band, row[EVID_COL])
#            else:
#                print "%s: rejected s for %d" % (short_band, row[EVID_COL])

            if accept_p_vert and accept_s_horiz:
                r = np.array((fit_p_vert[FIT_B], fit_s_horiz[FIT_B]))
                if lsp == None:
                    lsp = r
                else:
                    lsp = np.vstack([lsp, r])

#        print "plotting", base_coda_dir, band, lp.shape, ls.shape, lsp.shape
        plot_scatter(lp, ls, lsp, base_coda_dir, band)



def merge_plots(base_coda_dir, bands):
    for (band_idx, band) in enumerate(bands):
        try:
            pdf_dir = os.path.join(base_coda_dir, bands[band_idx][19:])

            try:
                os.remove(os.path.join(pdf_dir, "everything.pdf"))
            except:
                pass

            cmd = "gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=%s -dBATCH `ls %s/*.pdf`" % (os.path.join(pdf_dir, "everything.pdf"), pdf_dir)
            print "running", cmd
            os.popen(cmd)
        except:
            print traceback.format_exc()
            continue


def predict_trace(cursor, ev, row, noise_floor, cm_p, cm_s):
    siteid = int(row[SITEID_COL])
    evid = int(ev[EV_EVID_COL])
    sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (evid, siteid)
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    other_arrivals = other_arrivals[:, 0]

    starttime = np.min(other_arrivals)- 30
    endtime = np.max(other_arrivals) + 150
    srate = 10
    npts = int(((endtime-starttime)*srate))
    stats = {"starttime_unix": starttime, "sampling_rate": srate, "npts":npts, "siteid": siteid}


    data = noise_floor * np.ones( (npts,) )

    phase = row[P_PHASEID_COL] if row[P_PHASEID_COL] > 0 else 1
    pred_arr_time = ev[EV_TIME_COL] + cm_p.netmodel.mean_travel_time(row[LON_COL], row[LAT_COL], row[DEPTH_COL], int(row[SITEID_COL])-1, phase-1) + cm_p.predict_peak_time( ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL] )
    pred_arr_offset = int((pred_arr_time - starttime) * srate)
    pred_arr_height = ev[EV_MB_COL] * cm_p.predict_peak_amp( ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL] )
    print "pred amp p", pred_arr_height
    pred_decay_rate = cm_p.predict_decay( ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL] )
    for i in range(pred_arr_offset, npts):
        t = (i - pred_arr_offset)/srate
        data[i] += max(0, pred_arr_height + t*pred_decay_rate)

    phase = row[S_PHASEID_COL] if row[S_PHASEID_COL] > 0 else 5
    pred_arr_time = ev[EV_TIME_COL] + cm_s.netmodel.mean_travel_time(row[LON_COL], row[LAT_COL], row[DEPTH_COL], int(row[SITEID_COL])-1, phase-1) + cm_s.predict_peak_time( ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL] )
    pred_arr_offset = int((pred_arr_time - starttime) * srate)
    pred_arr_height = ev[EV_MB_COL] * cm_s.predict_peak_amp( ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL] )
    print "pred amp s", pred_arr_height
    pred_decay_rate = cm_s.predict_decay( ev, CodaModel.MODEL_TYPE_GAUSSIAN, row[DISTANCE_COL] )
    for i in range(pred_arr_offset, npts):
        t = (i - pred_arr_offset)/srate
        data[i] += max(0, pred_arr_height + t*pred_decay_rate)

    return Trace(data=data, header=stats)


def main():

    parser = OptionParser()


    parser.add_option("-s", "--siteid", dest="siteid", default=None, type="int", help="siteid of station for which to generate plots")
    parser.add_option("-f", "--full", dest="full", default=False, action="store_true", help="use the full coda window, all the way down to the noise floor (False)")
    parser.add_option("-r", "--runid", dest="runid", default=None, type="int", help="runid of coda fits to examine")
    parser.add_option("-b", "--basedir", dest="basedir", default=None, type=str, help="")

    parser.add_option("--scatter", dest="scatter", default=False, action="store_true", help="create scatter plots (False)")
    parser.add_option("--events", dest="events", default=False, action="store_true", help="(re)creates individual event coda plots (False)")
    parser.add_option("--pred_events", dest="pred_events", default=False, action="store_true", help="predicts individual event coda plots (False)")
    parser.add_option("--merge", dest="merge", default=False, action="store_true", help="merge all available plots for each band (False)")

    (options, args) = parser.parse_args()

    cursor = db.connect().cursor()

    if options.basedir is None:
        siteid = options.siteid
        full_label = "full" if options.full else "notfull"
        runid = options.runid

        base_coda_dir = get_base_dir(int(siteid), full_label, int(runid))
    else:
        base_coda_dir = options.basedir
    fname = os.path.join(base_coda_dir, 'all_data')
    all_data, bands = read_shape_data(fname)
    all_data = add_depth_time(cursor, all_data)
#    print fname
#    print bands
    print "read data", all_data.shape

    if options.scatter:
        generate_scatter_plots(all_data, bands, base_coda_dir)


    if options.events:
        for row in all_data:
            if int(row[EVID_COL]) != 5418898:
                continue

            band = bands[int(row[BANDID_COL])]
            short_band = band[19:]
            vert_noise_floor = row[VERT_NOISE_FLOOR_COL]
            horiz_noise_floor = row[HORIZ_NOISE_FLOOR_COL]

            (arrival_segment, noise_segment, other_arrivals, other_arrival_phases) = load_signal_slice(cursor, row[EVID_COL], row[SITEID_COL], load_noise = False)


            vert_smoothed, horiz_smoothed = smoothed_traces(arrival_segment, band)

            fit_p_vert = fit_from_row(row, P=True, vert=True)
            fit_p_horiz = fit_from_row(row, P=True, vert=False)
            fit_s_vert = fit_from_row(row, P=False, vert=True)
            fit_s_horiz = fit_from_row(row, P=False, vert=False)

            accept_p_vert = accept_fit(fit_p_vert, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)
            accept_p_horiz = accept_fit(fit_p_horiz, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)
            accept_s_vert = accept_fit(fit_s_vert, min_coda_length=min_s_coda_length, max_avg_cost = avg_cost_bound)
            accept_s_horiz = accept_fit(fit_s_horiz, min_coda_length=min_s_coda_length, max_avg_cost = avg_cost_bound)


            pdf_dir = get_dir(os.path.join(base_coda_dir, short_band))
            pp = PdfPages(os.path.join(pdf_dir, str(int(row[EVID_COL])) + "_log.pdf"))

            title = "%s evid %d siteid %d mb %f \n dist %f azi %f \n p_b %f p_acost %f p_len %f \n s_b %f s_acost %f s_len %f " % (short_band, row[EVID_COL], row[SITEID_COL], row[MB_COL], row[DISTANCE_COL], row[AZI_COL], row[VERT_P_FIT_B], row[VERT_P_FIT_AVG_COST], row[VERT_P_FIT_CODA_LENGTH], row[HORIZ_S_FIT_B], row[HORIZ_S_FIT_AVG_COST], row[HORIZ_S_FIT_CODA_LENGTH])


            try:
                plot_channels(pp, vert_smoothed, vert_noise_floor, [], [], horiz_smoothed, horiz_noise_floor, [], [], all_det_times = other_arrivals, all_det_labels = other_arrival_phases, title = "")

#                plot_channels(pp, vert_smoothed, vert_noise_floor, [fit_p_vert, fit_s_vert], ["g-" if accept_p_vert else "r-", "g-" if accept_s_vert else "r-"], horiz_smoothed, horiz_noise_floor, [fit_p_horiz, fit_s_horiz], ["g-" if accept_p_horiz else "r-", "g-" if accept_s_horiz else "r-"], all_det_times = other_arrivals, all_det_labels = other_arrival_phases, title = title)
            except:
                print "error plotting:"
                print traceback.format_exc()
            finally:
                pp.close()



    if options.pred_events:



        pv_data = dict()
        ph_data = dict()
        sv_data = dict()
        sh_data = dict()
        for (band_idx, band) in enumerate(bands):
            band_data = extract_band(all_data, band_idx)
            pv_data[band] = clean_points(band_data, True, True)
            ph_data[band] = clean_points(band_data, True, False)
            sv_data[band] = clean_points(band_data, False, True)
            sh_data[band] = clean_points(band_data, False, False)

        for row in all_data:

            band = bands[int(row[BANDID_COL])]
            short_band = band[19:]
            vert_noise_floor = row[VERT_NOISE_FLOOR_COL]
            horiz_noise_floor = row[HORIZ_NOISE_FLOOR_COL]

            fit_p_vert = fit_from_row(row, P=True, vert=True)
            fit_p_horiz = fit_from_row(row, P=True, vert=False)
            fit_s_vert = fit_from_row(row, P=False, vert=True)
            fit_s_horiz = fit_from_row(row, P=False, vert=False)
            accept_p_vert = accept_fit(fit_p_vert, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)
            accept_p_horiz = accept_fit(fit_p_horiz, min_coda_length=min_p_coda_length, max_avg_cost = avg_cost_bound)
            accept_s_vert = accept_fit(fit_s_vert, min_coda_length=min_s_coda_length, max_avg_cost = avg_cost_bound)
            accept_s_horiz = accept_fit(fit_s_horiz, min_coda_length=min_s_coda_length, max_avg_cost = avg_cost_bound)

            # need to fabricate vert_smoothed and horiz_smoothed. each will be at the noise level until the predicted arrival time, then will

            try:
                ev = row_to_ev(cursor, row)
                cm_pv = CodaModel(pv_data[band], "", True, True, ignore_evids = (int(row[EVID_COL]),), w1 = [0.00001, 0.01, 0.01], w2 = [25, 25, 25], s2 = [0.01, 6, 2])
                cm_ph = CodaModel(ph_data[band], "", True, False, ignore_evids = (int(row[EVID_COL]),), w1 = [0.00001, 0.01, 0.01], w2 = [25, 25, 25], s2 = [0.01, 6, 2])
                cm_sv = CodaModel(sv_data[band], "", False, True, ignore_evids = (int(row[EVID_COL]),), w1 = [0.00001, 0.01, 0.01], w2 = [25, 25, 25], s2 = [0.01, 6, 2])
                cm_sh = CodaModel(sh_data[band], "", False, False, ignore_evids = (int(row[EVID_COL]),), w1 = [0.00001, 0.01, 0.01], w2 = [25, 25, 25], s2 = [0.01, 6, 2])

                vert_predicted = predict_trace(cursor, ev, row, vert_noise_floor, cm_pv, cm_sv)
                horiz_predicted = predict_trace(cursor, ev, row, horiz_noise_floor, cm_ph, cm_sh)

                (b_col, gen_decay, gen_onset, gen_amp) = construct_output_generators(cursor, cm_pv.netmodel, False, False)

                amp = gen_amp(row)
                print "gen amp", amp

                print row[HORIZ_S_FIT_PEAK_HEIGHT] - horiz_noise_floor, row[MB_COL]

                print row[HORIZ_S_FIT_PEAK_HEIGHT]

            except ValueError:
                continue

            pdf_dir = get_dir(os.path.join(base_coda_dir, short_band))
            pp = PdfPages(os.path.join(pdf_dir, str(int(row[EVID_COL])) + "_pred.pdf"))
            title = "PREDICTED %s evid %d siteid %d mb %f \n dist %f azi %f \n p_b %f p_acost %f p_len %f \n s_b %f s_acost %f s_len %f " % (short_band, row[EVID_COL], row[SITEID_COL], row[MB_COL], row[DISTANCE_COL], row[AZI_COL], row[VERT_P_FIT_B], row[VERT_P_FIT_AVG_COST], row[VERT_P_FIT_CODA_LENGTH], row[HORIZ_S_FIT_B], row[HORIZ_S_FIT_AVG_COST], row[HORIZ_S_FIT_CODA_LENGTH])

            try:
                plot_channels(pp, vert_predicted, vert_noise_floor, [fit_p_vert, fit_s_vert], ["g-" if accept_p_vert else "r-", "g-" if accept_s_vert else "r-"], horiz_predicted, horiz_noise_floor, [fit_p_horiz, fit_s_horiz], ["g-" if accept_p_horiz else "r-", "g-" if accept_s_horiz else "r-"], title = title)
            except:
                print "error plotting:"
                print traceback.format_exc()
            finally:
                pp.close()


    if options.merge:
        merge_plots(base_coda_dir, bands)

if __name__ == "__main__":
    main()


