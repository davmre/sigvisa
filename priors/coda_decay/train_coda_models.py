
from database.dataset import *
from database import db

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.backends.backend_pdf import PdfPages

import sys, os, pickle

import utils.geog
import obspy.signal.util
import learn, sigvisa_util

import numpy as np
import scipy.linalg
import hashlib

from priors.coda_decay.coda_decay_common import *
from utils.gp_regression import GaussianProcess


def pred_arrtime(cursor, r, netmodel, phaseid_col, phase_arr_time_col):
    cursor.execute("select time, depth from leb_origin where evid=%d" % (r[EVID_COL]))
    (t, d) = cursor.fetchone()
    pred_arr_time = t + netmodel.mean_travel_time(r[LON_COL], r[LAT_COL], d, int(r[SITEID_COL])-1, int(r[phaseid_col])-1)

    sql_query="SELECT l.time, l.arid FROM leb_arrival l , static_siteid sid, leb_origin lebo, leb_assoc leba where lebo.evid=%d and lebo.orid=leba.orid and leba.arid=l.arid and sid.sta=l.sta and sid.id=%d order by l.time" % (r[EVID_COL], r[SITEID_COL])
    cursor.execute(sql_query)
    other_arrivals = np.array(cursor.fetchall())
    other_arrivals = other_arrivals[:, 0]
    start_time = np.min(other_arrivals) - 30

    return (pred_arr_time - start_time) - r[phase_arr_time_col]

def learn_models(band_data, P, vert, earthmodel, gen_target_col, w, s2, pp, label):

    clean_data = clean_points(band_data, P=True, vert=True)
    n = clean_data.shape[0]
    Xd = clean_data[:, [LAT_COL, LON_COL, DEPTH_COL]]
    X = clean_data[:, [LAT_COL, LON_COL]]
    y = np.zeros((n,))
    for i in range(n):
        y[i] = gen_target_col(clean_data[i,:])
    ll_kernel = lambda  ll1, ll2 : np.exp(-1 * utils.geog.dist_km(ll1, ll2)**2 / (w**2))
    lld_kernel = lambda  lld1, lld2 : np.exp(-1 * ( utils.geog.dist_km(lld1[0:2], lld2[0:2])**2 + np.abs(lld1[2] - lld2[2])**2 )   / (w**2)   )

#    llt_kernel = lambda  lld1, lld2 : np.exp(-1 * .5 * ( earthmodel.ArrivalTimeCoord(lld1[0], lld1[1], lld1[2], 0, 0 if P else 4, lld2[0], lld2[1], lld2[2]) + earthmodel.ArrivalTimeCoord(lld2[0], lld2[1], lld2[2], 0, 0 if P else 4, lld1[0], lld1[1], lld1[2])) ** 2 / (w ** 2))

    gp = GaussianProcess(ll_kernel, s2)
    gp.train(X, y)
    gpd = GaussianProcess(lld_kernel, s2)
    gpd.train(Xd, y)
#    gpt = GaussianProcess(llt_kernel, s2)
#    gpt.train(Xd, y)
    gpt = None

    regional_dist = []
    regional_y = []
    tele_dist = []
    tele_y = []
    for i in range(n):
        d = clean_data[i, DISTANCE_COL]
        dy = gen_target_col(clean_data[i,:])
        if d > 1000:
            tele_dist.append(d)
            tele_y.append(dy)
        else:            
            regional_dist.append(d)
            regional_y.append(dy)

    regional_model = utils.LinearModel.LinearModel("regional", ["distance"],
                                        [regional_dist,],
                                        regional_y)
    tele_model = utils.LinearModel.LinearModel("tele", ["distance"],
                                        [tele_dist,],
                                        tele_y)
    
    plt.figure()
    plt.title(label + " linear")
    plt.xlabel("distance (km)")
    plt.ylabel("")
    t = np.linspace(0, 1000, 50)
    pred = [ regional_model[tv] for tv in t ] 
    plt.plot(t, pred, "k-")
    t = np.linspace(1000, np.max(tele_dist), 50)
    pred = [ tele_model[tv] for tv in t ] 
    plt.plot(t, pred, "k-")
    plt.plot(np.concatenate([regional_dist, tele_dist]), np.concatenate([regional_y, tele_y]), 'ro')
    pp.savefig()

    regional_mean = np.mean(regional_y)
    regional_var = np.var(regional_y)
    tele_mean = np.mean(tele_y)
    tele_var = np.var(tele_y)

    plt.figure()
    plt.title(label + " regional gaussian mean %f sigma")
    n, bins, patches = plt.hist(regional_y, normed=1)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    y = mlab.normpdf( bincenters, regional_mean, np.sqrt(regional_var))
    plt.plot(bincenters, y, 'r--', linewidth=1)
    pp.savefig()
    
    plt.figure()
    plt.title(label + " tele gaussian")
    n, bins, patches = plt.hist(tele_y, normed=1)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    y = mlab.normpdf( bincenters, tele_mean, np.sqrt(tele_var))
    plt.plot(bincenters, y, 'r--', linewidth=1)
    pp.savefig()

    m = dict()
    m["gp"] = gp
    m["gpd"] = gpd
    m["gpt"] = gpt
    m["regional_linear"] = regional_model
    m["tele_linear"] = tele_model
    m["regional_gaussian"] = (regional_mean, regional_var)
    m["tele_gaussian"] = (tele_mean, tele_var)
    return m

class CodaModel:

    MODEL_TYPE_GP, MODEL_TYPE_GPD, MODEL_TYPE_GPT, MODEL_TYPE_LINEAR, MODEL_TYPE_GAUSSIAN = range(5)

    def __init__(self, basedir, ignore_evids = None):
        cursor = db.connect().cursor()
        sites = read_sites(cursor)
        st  = 1237680000
        et = st + 3600*24
        site_up = read_uptime(cursor, st, et)
        detections, arid2num = read_detections(cursor, st, et, arrival_table="leb_arrival", noarrays=True)
        phasenames, phasetimedef = read_phases(cursor)
        earthmodel = learn.load_earth("parameters", sites, phasenames, phasetimedef)
        netmodel = learn.load_netvisa("parameters", st, et, detections, site_up, sites, phasenames, phasetimedef)

        fname = os.path.join(basedir, 'all_data')
        all_data, bands = read_shape_data(fname)

        print "filtering", all_data.shape
        if ignore_evids is not None:
            good_rows = np.array([int(all_data[i, EVID_COL]) not in ignore_evids for i in range(all_data.shape[0])])
            all_data_n = all_data[good_rows, :]
            all_data = all_data_n
        print "to", all_data.shape

        all_data = add_depth_time(cursor, all_data)

        self.siteid = int(all_data[0, SITEID_COL])
        self.slon = sites[self.siteid-1, 0]
        self.slat = sites[self.siteid-1, 1]

        self.decay_models = dict()
        self.peak_amp_models = dict()
        self.onset_models = dict()

        for (band_idx, band) in enumerate(bands):

            outfile = os.path.join(basedir, band[19:], 'model_fits.pdf')
            pp = PdfPages(outfile)
            print "saving plots to", outfile

            band_data = extract_band(all_data, band_idx)
            
            self.decay_models[band] = []
            self.peak_amp_models[band] = []
            self.onset_models[band] = []

            for P in (True, False):
                for vert in (True, False):
                    idx = (0 if P else 2) + (0 if vert else 1)

                    if P and vert:
                        b_col = VERT_P_FIT_B
                    elif P and not vert:
                        b_col = HORIZ_P_FIT_B
                    elif not P and vert:
                        b_col = VERT_S_FIT_B
                    elif not P and not vert:
                        b_col = HORIZ_S_FIT_B

                    # learn decay rate models
                    gen_target_col = lambda x :x[b_col]
                    m = learn_models(band_data, P, vert, earthmodel, gen_target_col, 50, 0.001, pp, "decay P=%s vert=%s" % (P, vert))
                    self.decay_models[band].append(m)

                    # learn onset time models
                    gen_target_col = lambda x : pred_arrtime(cursor, x, netmodel, P_PHASEID_COL if P else S_PHASEID_COL, b_col + (VERT_P_FIT_PEAK_OFFSET - VERT_P_FIT_B))
                    m = learn_models(band_data, P, vert, earthmodel, gen_target_col, 50, 0.1, pp, "offset P=%s vert=%s" % (P, vert))
                    self.onset_models[band].append(m)

                    # learn peak height models
                    gen_target_col = lambda x : x[b_col + (VERT_P_FIT_HEIGHT - VERT_P_FIT_B)] - x[b_col] * (x[b_col + (VERT_P_FIT_CODA_START_OFFSET - VERT_P_FIT_B)] - x[b_col + (VERT_P_FIT_PEAK_OFFSET - VERT_P_FIT_B)])
                    m = learn_models(band_data, P, vert, earthmodel, gen_target_col, 50, 0.5, pp, "amp P=%s vert=%s" % (P, vert))
                    self.peak_amp_models[band].append(m)
                    
            pp.close()
            

    def predict(self, ev, band, P, vert, model_type, model_set, distance = None):
        idx = (0 if P else 2) + (0 if vert else 1)

        if distance is None:
            distance = utils.geog.dist_km((ev[EV_LON_COL], ev[EV_LAT_COL]), (slon, slat))

        if model_type == MODEL_TYPE_GP:
            return self.model_set[band][idx].gp.predict((ev[EV_LON_COL], ev[EV_LAT_COL]))
        elif model_type == MODEL_TYPE_GPD:
            return self.model_set[band][idx].gpd.predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == MODEL_TYPE_GPT:
            return self.model_set[band][idx].gpt.predict((ev[EV_LON_COL], ev[EV_LAT_COL], ev[EV_DEPTH_COL]))
        elif model_type == MODEL_TYPE_LINEAR:
            if distance < 1000:
                model = self.model_set[band][idx].regional_linear
            else:
                model = self.model_set[band][idx].tele_linear
            return model[distance]
        elif model_type == MODEL_TYPE_GAUSSIAN:
            if distance < 1000:
                (mean, var) = self.model_set[band][idx].regional_gaussian
            else:
                (mean, var) = self.model_set[band][idx].tele_gaussian
            return mean

    def predict_decay(self, ev, band, P, vert, model_type, distance = None):
        return self.predict(ev, band, P, vert, model_type, self.decay_models, distance=distance)

    def predict_peak_time(self, ev, band, P, vert, model_type, distance = None):
        return self.predict(ev, band, P, vert, model_type, self.onset_models, distance=distance)

    def predict_peak_amp(self, ev, band, P, vert, model_type, distance = None):
        return self.predict(ev, band, P, vert, model_type, self.peak_amp_models, distance=distance)



def main():
    cm = CodaModel(sys.argv[1], ignore_evids = (5330781,))

if __name__ == "__main__":
    main()
