from sigvisa import Sigvisa
from sigvisa.models.ttime import tt_predict
from sigvisa.utils.fileutils import mkdir_p
from sigvisa.treegp.gp import GPCov, GP, prior_sample, optimize_gp_hyperparams
from sigvisa.models.distributions import LogNormal
from sigvisa.plotting.event_heatmap import EventHeatmap
from sigvisa.source.event import Event
from sigvisa.utils.geog import dist_km
from sigvisa.signals.io import fetch_waveform, MissingWaveform
import scipy.stats
import pywt
import scipy.weave as weave
from scipy.weave import converters
from sigvisa.explore.doublets.xcorr_pairs import xcorr, xcorr_valid
import numpy as np
import pylab as plt
import os
import time
from sigvisa.explore.wavelet_vis import *

"""
stas = ['MKAR', 'TORD', 'ASF', 'MMAI', 'KEST', 'VRAC', 'AKASG', 'EIL', 'BRTR', 'GERES']

ev_bounds = dict()
ev_bounds['top'] = 50
ev_bounds['bottom'] = 30
ev_bounds['left'] = 20
ev_bounds['right'] =40

phase = 'P'
"""

ev_bounds = dict()

ev_bounds['top'] = 50
ev_bounds['bottom'] = 35
ev_bounds['left'] = 60
ev_bounds['right'] =100

basedir = "vdec_signals"
phase= "Pn"

def select_events(train=True):
    lbl = 'train' if train else 'test'
    try:
        r = np.load('explore/toy_wavelets/events_%s.npy' % lbl)
    except IOError:
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        if train:
            stime = 1238889600
            etime = 1245456000
        else:
            stime = 1237680000
            etime = 1238284800
        sql_query = "select lon,lat,0,time,evid from leb_origin where lon between %f and %f and lat between %f and %f and time between %d and %d and mb > 2.5" % (ev_bounds['left'], ev_bounds['right'], ev_bounds['bottom'], ev_bounds['top'], stime, etime)
        cursor.execute(sql_query)
        r = np.array(cursor.fetchall())
        mkdir_p('explore/toy_wavelets/')
        np.save('explore/toy_wavelets/events_%s.npy' % lbl, r)

    return r

def flatten_wavelet_coefs(coefs, levels=3):
    return np.concatenate([coefs[i] for i in range(min(levels, len(coefs)))])

def unflatten_wavelet_coefs(cc, wtype="db9", len_s=600):
    coefs = []

    cc = np.asarray(cc)

    filter_len = pywt.Wavelet(wtype).dec_len
    levels = pywt.dwt_max_level(len_s, filter_len)
    level_sizes = []
    k = len_s
    for i in range(levels):
        k = int(np.ceil(k/float(2)))
        level_sizes.append(k)
    level_sizes.append(k)
    level_sizes.append(0)
    level_sizes.reverse()
    bounds = np.cumsum(level_sizes)
    # wtype db9, len_s 600 should yield
    # bounds = [  0  19  38  76 151 301 601]

    for i in range(len(bounds)-1):
        if len(cc) >= bounds[i+1]:
            ncc = cc[bounds[i]:bounds[i+1]]
        else:
            ncc = np.zeros((bounds[i+1] - bounds[i]))
        coefs.append(ncc.flatten())
    return coefs

def gen_signals_sta(y):
    n_evs = len(y['tt_residual'])
    signals = np.zeros((n_evs, 1200))

    for i in range(n_evs):
        coefs_flat = [y[w][i] for w in sorted(y.keys()) if w.startswith('w')]
        coefs = unflatten_wavelet_coefs(coefs_flat)
        signal = pywt.waverec(coefs, 'db2', 'per')
        noise = np.random.randn(1200,)
        signals[i,:] = noise
        signals[i,400:800] += signal * 2
    return signals

def sample_synthetic_data_sta(wavelet_feature_names, X_events, sta):
    atime_cov = GPCov(dfn_str="lld", wfn_str="compact2", dfn_params=[200.0, 100.0], wfn_params=[4.0,])
    wiggle_cov = GPCov(dfn_str="lld", wfn_str="compact2", dfn_params=[40.0, 100.0], wfn_params=[100.0,])

    y = dict()
    gp_residuals = prior_sample(X_events, atime_cov, noise_var=0.0)
    y['tt_residual'] = gp_residuals + np.random.randn(X_events.shape[0]) * 3.0
    times = X_events[:, 3]
    evs = [Event(lon=x[0], lat=x[1], depth=x[2], time=x[3], mb=4.0) for x in X_events]
    pred_tts = np.array([tt_predict(ev, sta, phase) for ev in evs])

    y['arrival_time_picked'] = times + pred_tts + y['tt_residual']
    y['arrival_time_true'] = times + pred_tts + gp_residuals

    for wavelet_feature in wavelet_feature_names:
        y[wavelet_feature] = prior_sample(X_events, wiggle_cov, noise_var=0.1)
    return y

def sample_synthetic_data(stas, wavelet_feature_names, X_events):
    ys = dict()
    for sta in stas:
        ys[sta] = sample_synthetic_data_sta(wavelet_feature_names, X_events, sta)
    return ys

def get_leb_atime(cursor, evid, sta, phase='Pn'):
    sql_query = "select l.time from leb_arrival l, leb_assoc leba, leb_origin lebo where lebo.evid=%d and leba.phase='%s' and l.sta='%s' and leba.orid=lebo.orid and l.arid=leba.arid" % (evid, phase, sta)
    cursor.execute(sql_query)
    try:
        r = cursor.fetchall()[0]
        return float(r[0])
    except:
        return np.nan

def plot_heat(ax, stas, ev_true, y, all_gps, n=20, logscale=True):
    ev_tmp = Event(lon=ev_true.lon, lat=ev_true.lat, depth=ev_true.depth, mb=ev_true.mb, time=ev_true.time)
    def f(lon, lat):
        ev_tmp.lon=lon
        ev_tmp.lat = lat
        if logscale:
            return joint_log_likelihood(ev_tmp, y, all_gps)
        else:
            return np.exp(joint_log_likelihood(ev_tmp, y, all_gps))

    hm = EventHeatmap(f=f, calc=True, n=n, top_lat=ev_bounds['top'], bottom_lat=ev_bounds['bottom'], left_lon=ev_bounds['left'], right_lon=ev_bounds['right'])
    hm.add_stations(stas)
    #hm.init_bmap(axes=ax, nofillcontinents=True)
    #hm.plot_earth()
    #hm.plot_density(smooth=True)
    hm.set_true_event(ev_true.lon, ev_true.lat)
    hm.plot(axes=ax, smooth=True)

def plot_param_values(ax, stas, X_events, y):
    hm = EventHeatmap(f=None, calc=False, top_lat=ev_bounds['top'], bottom_lat=ev_bounds['bottom'], left_lon=ev_bounds['left'], right_lon=ev_bounds['right'])
    hm.add_stations(stas)
    hm.add_events(X_events, yvals=y)
    hm.plot(axes=ax, event_alpha=0.8, colorbar=True, offmap_station_arrows=False, label_stations=True)

def plot_traintest(ax, stas, X_train, X_test):
    hm = EventHeatmap(f=None, calc=False, top_lat=ev_bounds['top'], bottom_lat=ev_bounds['bottom'], left_lon=ev_bounds['left'], right_lon=ev_bounds['right'])
    hm.add_stations(stas)
    hm.init_bmap(axes=ax, nofillcontinents=True)
    hm.plot_earth()
    hm.plot_locations([(x[0], x[1]) for x in X_train], marker='.', ms=6, mec="none", mew=0, alpha=0.1)
    hm.plot_locations([(x[0], x[1]) for x in X_test], marker='.', ms=12, mec="none", mew=0, alpha=1.0, c='red')

def plot_posterior(ax, stas, ev_true, samples, width_deg=None, X_train=None, alphas="decay", plot_mean=True, marker_size=6, train_marker_size=4, **kwargs):

    if width_deg is None:
        width_deg = np.max(np.max(samples[:,:2], axis=0) - np.min(samples[:,:2], axis=0))
        print "inferring plot width of ", width_deg, "degrees"

    hm = EventHeatmap(f=None, calc=False, center=(ev_true.lon, ev_true.lat), width_deg=width_deg, height_deg=width_deg)

    hm.add_stations(stas)
    #hm.add_events(samples, )
    #hm.set_true_event(ev_true.lon, ev_true.lat)

    hm.plot(axes=ax, label_stations=True, **kwargs)

    if alphas == "decay":
        a = 1.0 / np.log(samples.shape[0] + 1 - np.arange(samples.shape[0]))
    elif alphas == "uniform":
        a = 1.0 / np.log(samples.shape[0] + 1)


    hm.plot_locations([(s[0], s[1]) for s in samples], marker='o', ms=6, mec="none", mew=0, alpha=a, c='red')

    hm.plot_locations([(ev_true.lon, ev_true.lat),], marker='+', ms=24, mec="blue", mew=5, alpha=1.0, c='blue')


    if X_train is not None:
        hm.plot_locations([(x[0], x[1]) for x in X_train], marker='x', ms=train_marker_size, mec="black", mew=train_marker_size/4, alpha=1.0, c='black')

    print "true ev:", ev_true
    means = np.mean(samples, axis=0)
    print "mean lon %.3f, lat %.3f, time %.3f" % (means[0], means[1], means[2])
    stds = np.std(samples, axis=0)
    print "stds: lon %.3f, lat %.3f, time %.3f" % (stds[0], stds[1], stds[2])

    hm.plot_locations([(means[0], means[1])], marker='8', ms=12, mec="purple", mew=3, alpha=1.0, c="purple")

    spatial_err = dist_km((means[0], means[1]), (ev_true.lon, ev_true.lat))
    time_err = np.abs(means[2] - ev_true.time)
    print "mean: spatial err %.4fkm time err %.4fs" % (spatial_err, time_err,)
    return hm

def train_GPs_sta(sta, X_events, y, atime_cov, atime_noise_var, wiggle_cov, wiggle_noise_var, obs_wiggle=True):
    gps = dict()

    nans = np.isnan(y['tt_residual'])
    ytt = y['tt_residual'][~nans]
    Xtt = X_events[~nans,:]
    gps['tt_residual'] = GP(X=Xtt, y=ytt, cov_main = atime_cov, noise_var=atime_noise_var)
    for k in y.keys():
        if not k.startswith('w'): continue
        kk = 'obs'+k if obs_wiggle else k
        yy = y[kk]
        nans = np.isnan(yy)
        yyy = yy[~nans]
        XX = X_events[~nans,:]
        gps[k] = GP(X=XX, y=yyy, cov_main = wiggle_cov, noise_var=wiggle_noise_var)
    return gps


def train_GPs_sta_opt(sta, X_events, y, atime_cov, atime_noise_var, wiggle_cov, wiggle_noise_var, test_evid, obs_wiggle=True, inferred_tts=True):
    gps = dict()

    ttr_key = 'tt_residual' + ('_inferred' if inferred_tts else '')

    nans = np.isnan(y[ttr_key])
    ytt = y[ttr_key][~nans]
    Xtt = X_events[~nans,:]
    #gps['tt_residual'] = GP(X=Xtt, y=ytt, cov_main = atime_cov, noise_var=atime_noise_var)
    for k in sorted(y.keys()):
        if k == "tt_residual":
            kk = k
        elif obs_wiggle:
            if not k.startswith('obsw'): continue
            kk = k
            k = k[3:]
        else:
            if not k.startswith('w'): continue
            kk = 'obs'+k
        yy = y[kk]
        nans = np.isnan(yy)
        yyy = yy[~nans]
        XX = X_events[~nans,:]

        nv_start = np.var(yyy)
        wc_new = wiggle_cov.copy()
        wc_new.wfn_params[0] = nv_start
        nllgrad, x0, bounds, build_gp, covs_from_vector = optimize_gp_hyperparams(noise_var=nv_start, noise_prior=LogNormal(0.0, 2.0), cov_main=wc_new, X=XX, y=yyy)

        def nllgrad_noise(n):
            n1, n2 = n
            x = np.copy(x0)
            x[0] = n1
            x[1] = n2
            ll, grad = nllgrad(x)
            return ll, grad[0:2]

        print 'optimizing noise for %s at %s, starting with %.5f' % (k, sta, nv_start)
        r = scipy.optimize.minimize(nllgrad_noise, (nv_start, nv_start), jac=True)
        print "got noise var, signal var", r.x
        nv = np.abs(r.x[0])
        wv = np.abs(r.x[1])
        wc_new.wfn_params[0] = wv
        gps[k] = GP(X=XX, y=yyy, cov_main = wc_new, noise_var=nv)

        mkdir_p(os.path.join(basedir, str(test_evid), 'gps', sta))
        fname = os.path.join(basedir, str(test_evid), 'gps', sta, k + '.gp')
        print "saved to", fname
        gps[k].save_trained_model(fname)

    return gps

def joint_log_likelihood_times_only(ev, y, all_gps):
    ll = 0
    X = np.array([[ev.lon, ev.lat, 0,],])
    for (sta, param_gps) in all_gps.items():
        try:
            pred_tt = tt_predict(ev, sta, phase)
        except ValueError as e:
            #print e, ev.lon, ev.lat, sta
            ll += -9999
            continue
        try:
            tt_residual = y[sta]['arrival_time_picked'] - (ev.time + pred_tt)
        except KeyError:
            continue
        ll += param_gps['tt_residual'].log_p(tt_residual, X, method='sparse', eps_abs=1e-8)
    return ll

def l2_error(x, std=1.0):
    n = len(x)
    code = """
    double ll = 0;
    for(int i=0; i < n; ++i) {
        double r = x(i)/std;
        ll += -.5 * r * r;
    }
    return_val = ll;
    """
    ll = weave.inline(code, ['x', 'n', 'std'],
                    type_converters=converters.blitz,
                    compiler='gcc')
    ll += -.5 * np.log(2 * np.pi * std * std) * n
    return ll

def get_ev_X(evs_sta, evid=None):

    evid_match = evs_sta[:, 4] == evid
    n = np.arange(len(evid_match))[evid_match][0]
    return evs_sta[n,:], n

def predict_signal(X, gps_sta, wtype="db9", wavelet_only=False, sample=False):
    wkeys = [k for k in sorted(gps_sta.keys()) if k.startswith('w')]

    if sample:
        coefs = [gps_sta[k].sample(X, method="sparse", include_obs=False, eps_abs=1e-8) for k in wkeys]
    else:
        coefs = [gps_sta[k].predict(X) for k in wkeys]
    #pred_ll = np.sum([gps_sta[k].log_p(coefs[i], X) for (i,k) in enumerate(wkeys)])
    if wavelet_only:
        return coefs

    cc = unflatten_wavelet_coefs(coefs, wtype=wtype)
    pred_wavelet = pywt.waverec(cc, wtype, 'per')
    return pred_wavelet


def joint_log_likelihood_full(ev, y, atimes, signals, all_gps, waves=True, times=True, only_stas=None, verbose=False, base_start=None, base_end=None, wtype=None):
    ll = 0
    X = np.array([[ev.lon, ev.lat, 0,],])

    if only_stas is None:
        only_stas = signals.keys()

    for sta in only_stas:
        param_gps = all_gps[sta]

        if times:
            pred_tt = tt_predict(ev, sta, phase)
            tt_residual = atimes[sta] - (ev.time + pred_tt)
            ll1 = param_gps['tt_residual'].log_p(tt_residual, X, include_obs=False)
            ll += ll1

            try:
                pick_error_ll = scipy.stats.norm.logpdf(y[sta]['arrival_time_picked'] - atimes[sta])
                ll2 = pick_error_ll
                ll += ll2
            except KeyError:
                # not all stations have picked arrival times for every event
                ll2 = 0

        #if verbose:
        #    print "sta %s tt_ll %.1f pick_ll %f tt_residual %.1f std %f" % (sta, ll1, ll2, tt_residual, np.sqrt(param_gps['tt_residual'].variance(X, include_obs=False)))

        if not waves:
            continue

        # need an arrival idx into the signal
        atime_offset = int((atimes[sta] - y[sta]['arrival_time_loaded']) * 20)

        # if atime_offset is negative and greater than base_start, then we miss the beginning of the signal
        obs_signal = signals[sta]
        obs_len = base_end-base_start
        true_start = max(0, base_start+atime_offset)
        true_end = min(len(obs_signal), base_end+atime_offset)
        missing_prefix = max(0, -true_start)
        missing_suffix = max(0, true_end - len(signals[sta]) )
        last_obs = obs_len - missing_suffix

        """
        wkeys = [k for k in sorted(param_gps.keys()) if k.startswith('w')]
        coefs = [param_gps[k].predict(X) for k in wkeys]
        pred_ll = np.sum([param_gps[k].log_p(coefs[i], X) for (i,k) in enumerate(wkeys)])
        cc = unflatten_wavelet_coefs(coefs, wtype=wtype)
        pred_wavelet = pywt.waverec(cc, wtype, 'per')
        pred_signal = np.zeros((1200,))
        ll += pred_ll
        if verbose:
            print "    pred ll %.1f" % pred_ll
        ll_wiggle = 0

        pred_signal[true_start:true_end] = pred_wavelet[missing_prefix:last_obs]

        norm_obs = np.linalg.norm(obs_signal)
        norm_pred = np.linalg.norm(pred_signal)
        #norm_std = np.std(obs_signal[:100]/norm_obs) * 40
        norm_std = 0.2
        #print sta, norm_std
        #norm_std = noise_std / norm_obs

        ll_wiggle += l2_error(pred_wavelet[:missing_prefix]/norm_pred, std=norm_std)
        ll_wiggle += l2_error(pred_wavelet[last_obs:]/norm_pred, std=norm_std)

        #if atime_offset > -base_start and atime_offset < (len(obs_signal) - base_end):
        #    pred_signal[base_start+atime_offset:base_end+atime_offset] = pred_wavelet
        #    ll_wiggle += l2_error(obs_signal[0:400])
            #ll_wiggle += np.sum([scipy.stats.norm.logpdf(t) for t in obs_signal[0:400]])
        #else:
            #ll_wiggle += np.sum([scipy.stats.norm.logpdf(t) for t in pred_wavelet])
        #    ll_wiggle += l2_error(pred_wavelet)

        try:
            norm_obs_signal = obs_signal/norm_obs
            r = ( norm_obs_signal - pred_signal/norm_pred )
            r0 = r[:true_start]
            r1 = r[true_start:true_end]
            r2 = r[true_end:]
            ll_wiggle_start = l2_error(r0, std=norm_std)
            ll_wiggle_mid = l2_error(r1, std=norm_std)
            ll_wiggle_end = l2_error(r2, std=norm_std)

            #ll_obs_mid = l2_error(norm_obs_signal[true_start:true_end], std=norm_std)
            #log_improvement_ratio = ll_wiggle_mid - ll_obs_mid
            #print norm_std, ll_wiggle_start, ll_wiggle_mid, ll_wiggle_end, ll_obs_mid, log_improvement_ratio

            #log_improvement_ratio = max(log_improvement_ratio, 0)
            #ll_adj = log_improvement_ratio * (len(obs_signal) - true_end)/ (true_end-true_start)


            # HACK: since we only model a finite-length wiggle, assume
            # that we'd get the same improvement ratio for the timesteps
            # *after* the modeled wiggle.
            ll_wiggle += ll_wiggle_start + ll_wiggle_mid + ll_wiggle_end # + ll_adj
        except Exception as e:
            print e
            import pdb; pdb.set_trace()


        ll += ll_wiggle
        if verbose:
            print "   wiggle %f" % ll_wiggle
        """


        #if atime_offset > -400 and atime_offset < 400:
        obs_signal = np.zeros((base_end-base_start,))


        if true_start <= obs_len and true_end >= 0:
            obs_signal[missing_prefix:last_obs] = signals[sta][true_start:true_end]

        coefs = flatten_wavelet_coefs(pywt.wavedec(obs_signal, wtype, 'per'), levels=4)
        coefs /= np.linalg.norm(coefs)
        coef_keys = [k for k in sorted(all_gps[sta].keys()) if k.startswith('w')]

        ll3 = 0
        if verbose:
            print "wavelets ",
        for (i,k) in enumerate(coef_keys):
            try:
                lll = all_gps[sta][k].log_p(coefs[i], X, method='sparse', eps_abs=1e-8)
            except Exception as e:
                print e
                import pdb; pdb.set_trace()
            if verbose:
                print "%d %.1f + " % (i, lll),
            if np.isnan(lll):
                import pdb; pdb.set_trace()
            ll3 += lll
        ll += ll3
        if verbose:
            print " = %.1f" % ll3
            #if sta == 'TORD':
            #    import pdb; pdb.set_trace()
        if np.isnan(ll):
            import pdb; pdb.set_trace()

    return ll

def gen_test_ev(X):
    return Event(lon=X[0], lat=X[1], depth=X[2], time=X[3], mb=4.0)

def sample_test_values(ev, all_gps):
    y = dict()
    X = np.array([[ev.lon, ev.lat, 0,],])
    for (sta, param_gps) in all_gps.items():
        y[sta] = dict()
        pred_tt = tt_predict(ev, sta, phase)
        sampled_tt_residual = param_gps['tt_residual'].sample(X, include_obs=False)
        pick_error =  np.random.randn() * np.sqrt(param_gps['tt_residual'].noise_var)
        y[sta]['arrival_time_picked'] = ev.time + pred_tt + sampled_tt_residual + pick_error
        y[sta]['arrival_time_true'] = ev.time + pred_tt + sampled_tt_residual
        for (k, gp) in param_gps.items():
            y[sta][k] =  gp.sample(X)
    return y


def MH_move_space(ev, y, atimes, signals, all_gps, old_ll, waves, std_ll = 0.05, base_start=None, base_end=None, wtype=None):
    orig_lon, orig_lat = ev.lon, ev.lat
    ev.lon = orig_lon + np.random.randn() * std_ll
    ev.lat = orig_lat + np.random.randn() * std_ll
    new_ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, waves=waves, base_start=base_start, base_end=base_end, wtype=wtype)

    log_u = np.log(np.random.rand())
    if new_ll - old_ll > log_u:
        return True, new_ll
    else:
        ev.lon = orig_lon
        ev.lat = orig_lat
        return False, old_ll

def MH_move_time(ev, y, atimes, signals, all_gps, old_ll, waves, std_s = 1.0, base_start=None, base_end=None, wtype=None):
    orig_time = ev.time
    ev.time = orig_time + np.random.randn() * std_s
    new_ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, waves=waves, base_start=base_start, base_end=base_end, wtype=wtype)

    log_u = np.log(np.random.rand())

    if new_ll - old_ll > log_u:
        return True, new_ll
    else:
        ev.time = orig_time
        return False, old_ll

def MH_move_atime(sta, ev, y, atimes, signals, all_gps, waves, std_s = 1.0, base_start=None, base_end=None, wtype=None):
    old_ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, only_stas=[sta,], waves=waves, base_start=base_start, base_end=base_end, wtype=wtype)
    orig_time = atimes[sta]
    atimes[sta] = orig_time + np.random.randn() * std_s
    new_ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, only_stas=[sta,], waves=waves, base_start=base_start, base_end=base_end, wtype=wtype)

    log_u = np.log(np.random.rand())

    if new_ll - old_ll > log_u:
        return True
    else:
        atimes[sta] = orig_time
        return False

def xcorr_dist(a, b):
    # return the
    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(a)))

    xc = np.correlate(a, b, 'valid')
    #if (xc < 0).any():
    #    xc -= np.min(xc)

    #xc = xc ** 6
    xc = np.exp(xc * 30)

    xc /= np.sum(xc)
    return xc

def sample_discrete(pmf):
    u = np.random.rand()
    cdf = np.cumsum(pmf)
    r = np.searchsorted(cdf, u, side='left')
    return r

def MH_move_atimeXC(sta, ev, y, atimes, signals, all_gps, waves, base_start=None, base_end=None, wtype=None):
    old_ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, only_stas=[sta,], waves=waves, base_start=base_start, base_end=base_end, wtype=wtype)
    orig_time = atimes[sta]
    orig_idx = int((orig_time - y[sta]['arrival_time_loaded']) * 20 + base_start)

    sta_gps = all_gps[sta]
    x = np.array(((ev.lon, ev.lat, 0.0),))
    coefs = [sta_gps[k].predict(x) for k in sorted(sta_gps.keys()) if k.startswith('w')]
    cc = unflatten_wavelet_coefs(coefs)
    pred_wavelet = pywt.waverec(cc, "db9", 'per') # * 2

    # if we're far enough out that the pred wavelet is all zeros,
    # don't even bother trying to correlate it (doing so would lead to
    # NaNs since it doesn't normalize properly).
    if np.sum(np.abs(coefs)) < 1e-10:
        return False

    d = xcorr_dist(signals[sta], pred_wavelet)

    xx = np.linspace(-10, 30, len(d))
    d_atime = np.exp(-(xx / 3.0)**2)
    d *= d_atime
    d /= np.sum(d)

    new_idx = sample_discrete(d)
    atimes[sta] = (new_idx - base_start) / 20.0 + y[sta]['arrival_time_loaded']
    lp_propose = np.log(d[new_idx])


    if orig_idx < 0 or orig_idx > len(d):
        lp_reverse = -np.inf
    else:
        lp_reverse = np.log(d[orig_idx])

    if np.isnan(lp_reverse):
        import pdb; pdb.set_trace()

    new_ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, only_stas=[sta,], waves=waves, base_start=base_start, base_end=base_end, wtype=wtype)

    #plt.figure()
    #plt.plot(d)
    #d1 = np.exp(d)
    #plt.plot(d1 / np.sum(d1))
    #plt.title("xc at %s: current %d %.4f proposed %d %.4f" % (sta, orig_idx, lp_reverse, new_idx, lp_propose))

    log_u = np.log(np.random.rand())
    if (new_ll + lp_reverse) - (old_ll + lp_propose) > log_u:
        print "atimeXC succeeded at %s: lps (%.1f + %.1f) - (%.1f + %.1f), atime old %.1f new %.1f true/pred %.1f" % (sta, new_ll, lp_reverse, old_ll, lp_propose, orig_time, atimes[sta], y[sta]['arrival_time_loaded'])
        return True
    else:
        #print "atimeXC rejected at %s: lps (%.1f + %.1f) - (%.1f + %.1f), atime old %.1f new %.1f true %.1f" % (sta, new_ll, lp_reverse, old_ll, lp_propose, orig_time, atimes[sta], y[sta]['arrival_time_loaded'])

        atimes[sta] = orig_time
        return False

#def MH_move_EVatimeXC(sta, ev, y, atimes, signals, all_gps, waves, base_start=None, base_end=None, wtype=None):



def grid_search(y, all_gps, time_window_s=300, n_space=10, n_time = 20):
    print y.keys()
    print y[y.keys()[0]].keys()
    first_atime = np.min([yy['arrival_time_loaded'] for yy in y.values()])
    print 'first atime', first_atime

    lons = np.linspace(ev_bounds['left'], ev_bounds['right'], n_space)
    lats = np.linspace(ev_bounds['bottom'], ev_bounds['top'], n_space)
    times = np.linspace(first_atime - time_window_s, first_atime, n_time)

    ev_tmp = Event(lon=0, lat=0, depth=0, time=0, mb=4.0)
    best_ev = None
    best_ll = -np.inf
    for lon in lons:
        for lat in lats:
            #print "gridsearching at %.1f, %.1f" % (lon, lat)
            for time in times:
                ev_tmp.lon = lon
                ev_tmp.lat = lat
                ev_tmp.time = time
                ll = joint_log_likelihood_times_only(ev_tmp, y, all_gps)
                if ll > best_ll:
                    print "gridsearch: new best ll", ll, "at", (lon, lat, time)
                    best_ll = ll
                    best_ev = (lon, lat, time)

    return best_ev, best_ll

def run_MH(y, all_gps, signals, true_ev, X_train, burnin = 2000, steps = 2000, x0=None, waves=False, base_start=400, base_end=800, wtype='db2', visualize=True, width_deg=1.5, sample_fname=None):
    if x0 is None:
        (lon0, lat0, time0), ll = grid_search(y, all_gps)
    else:
        lon0, lat0, time0 = x0
        ll = np.float('-inf')

    ev = Event(lon=true_ev.lon, lat=true_ev.lat, depth=0, time=true_ev.time, mb=4.0)
    atimes = dict([(sta, y[sta]['arrival_time_loaded']) for sta in signals.keys()])
    ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, waves, base_start=base_start, base_end=base_end, wtype=wtype, verbose=True)

    print "true ev", true_ev
    print "log-likelihood for true ev with loaded atimes", ll

    samples = np.reshape(np.array([ev.lon, ev.lat, ev.time] + [atimes[k] for k in sorted(atimes.keys())]), (1,-1))

    if visualize:
        visualize_MCMC_state(true_ev, samples[0:1,:], y, signals, all_gps, X_train, wtype=wtype, base_start=base_start, base_end=base_end, waves=waves, width_deg=width_deg, offmap_station_arrows=False)

    ev = Event(lon=lon0, lat=lat0, depth=0, time=time0, mb=4.0)
    atimes = dict([(sta, y[sta]['arrival_time_loaded']) for sta in signals.keys()])
    print "intialized from grid search:", ev
    print "init: ll %f lon %.1f lat %.1f time %.1f" % (ll, ev.lon, ev.lat, ev.time,)
    samples = np.reshape(np.array([ev.lon, ev.lat, ev.time] + [atimes[k] for k in sorted(atimes.keys())]), (1,-1))

    #atimes = {"AAK": y['AAK']['arrival_time_loaded'] - 15.0/20,
    #          "AKTO": y['AKTO']['arrival_time_loaded'] - 2.0/20,
    #          "BVAR": y['BVAR']['arrival_time_loaded'] - 36.0/20,
    #          "MKAR": y['MKAR']['arrival_time_loaded'] - 4.0/20,
    #          "ZALV": y['ZALV']['arrival_time_loaded'] - 14.0/20}

    n = burnin + steps
    samples = np.zeros((steps, 3 + len(signals.keys())))
    accepts_space = np.zeros((n,))
    accepts_space_small = np.zeros((n,))
    accepts_time = np.zeros((n,))
    accepts_atime = np.zeros((n,))
    accepts_atimeXC = np.zeros((n/10,))

    t1 = time.time()
    for i in range(n):
        accepted, ll = MH_move_space(ev, y, atimes, signals, all_gps, ll, waves, base_start=base_start, base_end=base_end, wtype=wtype)
        accepts_space[i] = int(accepted)

        accepted, ll = MH_move_space(ev, y, atimes, signals, all_gps, ll, waves, base_start=base_start, base_end=base_end, wtype=wtype, std_ll=0.01)
        accepts_space_small[i] = int(accepted)

        accepted, ll = MH_move_time(ev, y, atimes, signals, all_gps, ll, waves, base_start=base_start, base_end=base_end, wtype=wtype)
        accepts_time[i] = int(accepted)

        for sta in signals.keys():
            accepted = MH_move_atime(sta, ev, y, atimes, signals, all_gps, waves, base_start=base_start, base_end=base_end, wtype=wtype)
            accepts_atime[i] = int(accepted)

            accepts_atimeXC[i/10] = 0
            if waves and (i % 10 == 0):
                accepted = MH_move_atimeXC(sta, ev, y, atimes, signals, all_gps, waves, wtype=wtype, base_start=base_start, base_end=base_end)
                accepts_atimeXC[i/10] += int(accepted) / float(len(signals.keys()))

        ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, waves, base_start=base_start, base_end=base_end, wtype=wtype)

        if i >= burnin:
            #print [atimes[sta] for sta in sorted(atimes.keys())]
            samples[i-burnin,:] = np.concatenate(((ev.lon, ev.lat, ev.time), [atimes[sta] for sta in sorted(atimes.keys())]))

        if i % 10 and sample_fname is not None:
            np.save(sample_fname, samples)


        if i % 10 == 0 or (i < 50 and i % 5 == 0):
            spatial_err = dist_km((ev.lon, ev.lat), (true_ev.lon, true_ev.lat))
            time_err = np.abs(ev.time - true_ev.time)
            print "step %d: ll %f lon %.1f lat %.1f time %.1f spatial err %.1fkm time err %.1fs acceptance rate space %f space_small %f time %f atime %f atimeXC %f" % (i, ll, ev.lon, ev.lat, ev.time, spatial_err, time_err, np.mean(accepts_space[:i]), np.mean(accepts_space_small[:i]), np.mean(accepts_time[:i]), np.mean(accepts_atime[:i]), np.mean(accepts_atimeXC[:i/10]))

            if i > 0:
                t2 = time.time()
                print "current: %f seconds/step. t1 %.3f t2 %.3f i %d" % ( (t2-t1)/float(i) , t1, t2, i)

            if visualize:
                visualize_MCMC_state(true_ev, samples[:i-burnin+1,:], y, signals, all_gps,
                                     X_train, base_start=base_start, base_end=base_end,
                                     wtype=wtype, waves=waves, width_deg=width_deg, offmap_station_arrows=False)
            ll = joint_log_likelihood_full(ev, y, atimes, signals, all_gps, waves, base_start=base_start, base_end=base_end, wtype=wtype)

    return ev, samples



def visualize_MCMC_state(ev_true, samples, y, signals, all_gps, X_train, width_deg=0.4, wtype='db2', base_start=400, base_end=800, waves=True, alphas="decay", scale=1.0,  **kwargs):

    def plot_pred_wiggle(ax, y_sta, signal, sta_gps, state, atime, txt):
        x = np.reshape(np.array([state[0], state[1], 0.0]), (1, -1))
        atime_offset = int((atime - y_sta['arrival_time_loaded'])*20)
        xx = np.linspace(-10, 50, 1200)
        sn  = signal/np.linalg.norm(signal)
        ax.plot(xx, sn, linewidth=0.5, color='black')

        #ax.bar(left=[(base_start + atime_offset) / 20.0 - 10,], height = np.max(sn) - np.min(sn), bottom=np.min(sn), width=.25, edgecolor='red', color='red', linewidth=1 )


        pred_wavelet = None
        if atime_offset > -800 and atime_offset < 800:
            #for i in range(5):
            coefs = np.array([sta_gps[k].predict(x) for k in sorted(sta_gps.keys()) if k.startswith('w')])
            cc = unflatten_wavelet_coefs(coefs, wtype=wtype)
            pred_wavelet = pywt.waverec(cc, wtype, 'per')
            assert(len(pred_wavelet) == base_end-base_start)
            xx = np.linspace((base_start+atime_offset) / 20.0 -10, (base_end+atime_offset)/20.0 - 10, base_end-base_start)
            ax.plot(xx, pred_wavelet / np.linalg.norm(pred_wavelet), linewidth=1.0, color='red')
        return pred_wavelet

        ax.set_title("%s" % (txt,))


    stas = signals.keys()
    plt.figure(figsize=(15*scale if waves else 7*scale, 7*scale))

    ax = plt.subplot2grid((5,6), (0,0), rowspan=5, colspan=3 if waves else 6)

    plot_posterior(ax, stas, ev_true, samples, width_deg=width_deg, X_train=X_train, alphas=alphas, **kwargs)

    if waves:
        colspan = 3 if len(signals.keys()) <= 5 else 1
        pred = dict()
        for (i, sta) in enumerate(sorted(signals.keys())):
            ax = plt.subplot2grid((5,6), (i % 5,3 + i/5), colspan=colspan)
            pred[sta] = plot_pred_wiggle(ax, y[sta], signals[sta], all_gps[sta], samples[-1,:], samples[-1,3+i], sta)
            print sta, samples[-1, 3+i], y[sta]['arrival_time_loaded'], samples[-1, 3+i] - y[sta]['arrival_time_loaded']
    plt.tight_layout()

    #for sta in stas:
    #    plt.figure()
    #    xc = xcorr_dist(pred[sta], signals[sta])
    #    plt.plot(xc)
    #    plt.title(sta)

def test_neighbors(X_train, X_test):
    for x in X_test:
        distances = np.array([dist_km((x[0], x[1]), (xx[0], xx[1])) for xx in X_train])
        print x, ': neighbors within'
        for threshold in (5, 15, 30, 100, 200, 1000):
            print '%dkm: %f' % (threshold, np.sum(distances < threshold))
            #print distances < threshold

def split_ys(ys, n):
    ys_train = dict()
    ys_test = dict()
    for sta in ys.keys():
        ys_train[sta] = dict()
        ys_test[sta] = dict()
        for k in ys[sta].keys():
            ys_train[sta][k] = np.array(ys[sta][k][:n])
            ys_test[sta][k] = np.array(ys[sta][k][n:])
    return ys_train, ys_test

def extract_event_ys(ys, n):
    ys_ev = dict()
    for sta in ys.keys():
        ys_ev[sta] = dict()
        for k in ys[sta].keys():
            ys_ev[sta][k] = np.array(ys[sta][k][n:n+1])
    return ys_ev

def plot_aligned(s1, s2):
    xcmax, offset = xcorr(s1, s2)
    x = np.arange(0, len(s1))
    plot(x, s1/np.linalg.norm(s1, 2))
    plot(x + offset, s2/np.linalg.norm(s2, 2))

def observe_rel_times(reftimes, xcs, offsets, xc_threshold=0.7, srate=20):
    xcs -= np.diag(np.diag(xcs))
    eyes, js = (xcs > xc_threshold).nonzero()
    pairs = (eyes, js)

    true_rel_times = reftimes[eyes] - reftimes[js]
    observed_rel_times = offsets[eyes, js] / float(srate) + true_rel_times
    return pairs, observed_rel_times

# the signals I see are 'aligned' to the true arrival times
# so we have to subtract out the true arrival times, i.e. an offset of 0 is really an
def align_atimes_sta(ys, pairs, observed_rel_times, pick_std, xc_std, sta, X_events=None, cov=None):

    evs = [Event(lon=x[0], lat=x[1], depth=x[2], time=x[3], mb=4.0) for x in X_events]
    print evs[0]
    print sta
    pred_tts = np.array([tt_predict(ev, sta, phase) for ev in evs])
    pred_ats = X_events[:,3] + pred_tts
    residuals = ys['arrival_time_picked'] - pred_ats

    def atime_energy(abs_errors):

        atimes = abs_errors + pred_ats
        pick_errors = atimes - ys['arrival_time_picked']

        eyes, js = pairs
        rel_atimes = atimes[eyes] - atimes[js]
        rel_errors = rel_atimes - observed_rel_times

        #abs_energy = np.sum((abs_errors / pick_std)**2)

        #
        #gp = GP(X=X_events,y=abs_errors, cov_main=cov, noise_var=noise_var, compute_ll=True)


        pick_energy = X_events.shape[0]*np.log(pick_std * np.sqrt(2*np.pi)) +  np.sum((pick_errors / pick_std)**2)
        grad = 2.0 / (pick_std*pick_std) * pick_errors

        if cov is not None:
            gp_energy, gpgrad = ll_under_GPprior(X_events, abs_errors, cov, 0.0)
            gp_energy *= -1
            grad -= gpgrad
        else:
            gp_energy = 0

        rel_energy = len(rel_errors)*np.log(xc_std * np.sqrt(2*np.pi)) +  np.sum((rel_errors / xc_std)**2)
        for (i, (eye, j)) in enumerate(zip(eyes, js)):
            grad[eye] += 2.0/(xc_std**2) * rel_errors[i]
            grad[j] -= 2.0/(xc_std**2) * rel_errors[i]

        return gp_energy+pick_energy+rel_energy, grad

    x0 = np.zeros(pred_tts.shape)


    r = scipy.optimize.minimize(atime_energy, x0, method='BFGS', jac=True, options={'disp': True})

    return r.x, pred_ats

def plot_signal_atime(signal, atime, true_atime):
    figure()
    plot(signal)
    offset = int((atime - true_atime) * 20)
    plt.bar(left=[400 + offset,], height = np.max(signal) - np.min(signal), bottom=np.min(signal), width=.25, edgecolor='red', color='red', linewidth=1 )

def extract_wavelets_sta(ys, signals, atimes, srate=20, base_start=200, base_end=800, wave_keys=None):
    if wave_keys is None:
        wave_keys = [k for k in sorted(ys.keys()) if k.startswith('w')]
    n_events = len(ys['arrival_time_picked'])
    for k in wave_keys:
        ys['obs'+k] = np.zeros((n_events,))

    for i in range(signals.shape[0]):
        atime_offset = int(np.round((atimes[i] - ys['arrival_time_picked'][i]) * srate))
        if atime_offset < -200:
            obs_signal = np.zeros((600,))
            obs_N = base_end+atime_offset
            if obs_N > 0:
                obs_signal[-obs_N:] = signals[i,:obs_N]
        elif atime_offset > 400:
            obs_signal = np.zeros((600,))
            obs_N = 1200 - (base_start+atime_offset)
            if obs_N > 0:
                obs_signal[:obs_N] = signals[i,-obs_N:]
        else:
            obs_signal = signals[i,base_start+atime_offset:base_end+atime_offset]

        assert(len(obs_signal) == base_end - base_start)
        coefs = flatten_wavelet_coefs(pywt.wavedec(obs_signal, 'db8', 'per'), levels=4)
        coefs /= np.linalg.norm(coefs, 2)

        for (j, k) in enumerate(wave_keys):
            ys['obs'+k][i] = coefs[j]
    return wave_keys


def wavelets_evid(evs_sta, ys_sta, evid):
    evid_match = evs_sta[:,4] == evid
    keys = [k for k in sorted(ys_sta.keys()) if k.startswith('obsw')]
    coefs = [ys_sta[k][evid_match][0] for k in keys]
    #cc = unflatten_wavelet_coefs(coefs, wtype="db9")
    return coefs






def load_vdec_signals(sta, basedir="vdec_signals"):
    ss = Sigvisa()
    s = np.load(os.path.join(ss.homedir, basedir, "signals_Pn_%s.npy" % sta))
    if s.shape[1] == 2407:
        ss = np.zeros((s.shape[0], 1207))
        ss[:,0:7] = s[:,0:7]
        ss[:,7:] = s[:,7::2]
        s = ss
    elif s.shape[1] != 1207:
        print "resampling from %d to 1200 at %s" % (s.shape[1], sta)
        ss = np.zeros((s.shape[0], 1207))
        ss[:,0:7] = s[:,0:7]
        for i in range(s.shape[0]):
            ss[i,7:] = scipy.signal.resample(s[i,7:], 1200)
        s = ss

    evs = s[:, :7]
    signals = s[:,7:]

    evids = np.copy(evs[:,0])
    evs[:, :4] = evs[:, 1:5]
    evs[:,4] = evids

    return evs, signals

def load_station_data(evids, sta):
    devids = pd.DataFrame([int(x) for x in evids], columns=['evid',])
    s = np.load(os.path.join(basedir, "signals_Pn_%s.npy" % sta))
    if s.shape[1] == 2407:
        ss = np.zeros((s.shape[0], 1207))
        ss[:,0:7] = s[:,0:7]
        ss[:,7:] = s[:,7::2]
        s = ss
    elif s.shape[1] != 1207:
        print "resampling from %d to 1200 at %s" % (s.shape[1], sta)
        ss = np.zeros((s.shape[0], 1207))
        ss[:,0:7] = s[:,0:7]
        for i in range(s.shape[0]):
            ss[i,7:] = scipy.signal.resample(s[i,7:], 1200)
        s = ss
    p = pd.DataFrame(s, columns=["evid", "lon", "lat", "depth", "time", "mb", "atime"] + ['%04d' % i for i in range(1200)])
    p['evid'] = p['evid'].astype(int)
    dd = pd.merge(p, devids, how='right', on='evid')

    evs = np.asarray(dd[["evid", "lon", "lat", "depth", "time", "mb", "atime"]].as_matrix())
    signals = np.asarray(dd[['%04d' % i for i in range(1200)]].as_matrix())

    return dd, evs, signals

def load_evids(stas):
    evids_once = set()
    evids_twice = set()
    for sta in stas:
        evs, signals = load_vdec_signals(sta)
        my_evids = set([int(evid) for evid in evs[:, 4]])
        already_seen = evids_once.intersection(my_evids)
        evids_twice.update(already_seen)
        evids_once.update(my_evids)

    print len(evids_once), len(evids_twice)
    return evids_twice, evs

def load_true_data_sta(sta):
    X_events, signals = load_vdec_signals(sta)

    y = dict()
    evs = [Event(lon=x[0], lat=x[1], depth=x[2], time=x[3], mb=4.0) for x in X_events]
    pred_atimes = np.array([ ev.time + tt_predict(ev, sta, 'Pn') for ev in evs])
    atimes = np.array([ x[6] for x in X_events])

    y['arrival_time_picked'] = atimes
    #y['tt_residual'] = pred_atimes - atimes

    #print 'got tt residuals', y['tt_residual']

    #features = fourier_features(signals)

    #wnames = ['f%03d' % i for i in range(features.shape[1])]
    #for (i, name) in enumerate(wnames):
    #    y[name] = features[:, i]

    return y

def load_true_data(stas):


    ys = dict()
    for sta in stas:
        print "loading data from", sta
        ys[sta] = load_true_data_sta(sta)


    return ys

#def find_doublets():


def do_align_atimes(stas, evs, signals, ys):
    atime_cov = GPCov(dfn_str="lld", wfn_str="compact2", dfn_params=[200.0, 100.0], wfn_params=[4.0,])

    pairs = dict()

    for sta in stas:

        try:
            fname = os.path.join(basedir, 'alignments_%s.npz' % sta)
            print "trying to load from", fname

            try:
                d = np.load(fname)
                dists = d['dists']
                xcs = d['xcs']
                offs = d['offs']
                aligned = d['aligned']
                inferred_atimes = d['inferred_atimes']
                eyes = d['eyes']
                js = d['js']
                print "loaded cross-correlation as", sta
            except Exception as e:
                print e
                print "computing cross-correlation at", sta
                dists = distance_pairs(evs[sta], evs[sta])
                xcs, offs = xcorr_pairs(signals[sta], signals[sta], dists, dist_threshold=40)

                print "inferring atimes at", sta
                pairs_sta, observed_rel_times = observe_rel_times(evs[sta][:,6], xcs, offs)

                #aligned, pred_ats = align_atimes_sta(ys[sta], pairs_sta, observed_rel_times, 3.0, 0.001, sta, evs[sta])
                aligned, pred_ats = align_atimes_sta(ys[sta], pairs_sta, observed_rel_times, 3.0, 0.001, sta, np.array(evs[sta], copy=True), atime_cov)

                inferred_atimes = pred_ats + aligned

                eyes, js = pairs_sta
                np.savez(fname, aligned=aligned, inferred_atimes=inferred_atimes,
                         xcs=xcs, offs=offs, dists=dists, eyes=eyes, js=js)


            ys[sta]['tt_residual_inferred'] = aligned
            ys[sta]['arrival_time_inferred'] = inferred_atimes
            pairs[sta] = (eyes, js)
        except Exception as e:
            print e
            continue
    return ys, pairs




# need to generate:
# a test event object
# a ys[] object with training data (but NOT the test event)
# a ys[] object with the test event: just picked arrival times and 'loaded' arrival times
# (this will be a new field: giving the times the loaded signals are aligned to. for synth datak this isthe true time; for real data, it can be the pred atime)

def isolate_test_ev(evs, ys, evid):
    new_evs = dict()
    new_ys = dict()

    test_ys = dict()
    ats_to_load = dict()
    ev = None

    for sta in ys.keys():
        evid_match = (evs[sta][:, 4] == evid)
        if np.sum(evid_match) == 1:
            x_ev = evs[sta][evid_match,:]
            ev = Event(lon=x_ev[0,0], lat=x_ev[0,1], depth=0, time=x_ev[0,3], mb=x_ev[0,5])
            print "got ev", ev
            break

    for sta in ys.keys():
        new_ys[sta] = dict()
        test_ys[sta] = dict()

        evid_match = (evs[sta][:, 4] == evid)
        ev_n = np.arange(len(evid_match))[evid_match]
        if len(ev_n) == 0:
            print "event not observed at", sta
            new_evs[sta] = evs[sta]
            new_ys[sta] = ys[sta]
        elif len(ev_n) == 1:
            print "event observed as %d at %s" % (ev_n, sta)
            new_evs[sta] = evs[sta][~evid_match,:]
            for k in ys[sta].keys():
                new_ys[sta][k] = ys[sta][k][~evid_match]
            test_ys[sta]['arrival_time_picked'] = ys[sta]['arrival_time_picked'][evid_match]

        else:
            raise Exception('multiple events at %s match evid %d' % (sta, evid))

        try:
            pred_at = ev.time + tt_predict(ev, sta, phase)
            ats_to_load[sta] = pred_at
            test_ys[sta]['arrival_time_loaded'] = pred_at
        except Exception as e:
            print "can't predict travel time to %s" % sta
            print e


    return new_evs, new_ys, test_ys, ats_to_load, ev


def train_all_gps_opt(evs, ys, test_evid):
    atime_cov = GPCov(dfn_str="lld", wfn_str="compact2", dfn_params=[200.0, 100.0], wfn_params=[3.0,])
    wiggle_cov = GPCov(dfn_str="lld", wfn_str="compact2", dfn_params=[40.0, 100.0], wfn_params=[0.01,])
    atime_noise_var = 9.0
    wiggle_noise_var = 0.1

    all_gps = dict()
    for (sta,y) in ys.items():
        try:
            X_events = evs[sta]
        except:
            X_events = evs

        try:
            all_gps[sta] = train_GPs_sta_opt(sta, X_events, y, atime_cov, atime_noise_var, wiggle_cov, wiggle_noise_var, test_evid)
        except Exception as e:
            print e
            print "ERROR TRAINING MODEL at", sta
            continue
    return all_gps

def save_all_gps(test_evid, all_gps):
    for (sta, gps) in all_gps.items():
        print "saving", sta
        for (key, gp) in gps.items():
            #print 'saving', sta, key
            mkdir_p(os.path.join(basedir, str(test_evid), 'gps', sta))
            fname = os.path.join(basedir, str(test_evid), 'gps', sta, key + '.gp')
            gp.save_trained_model(fname)

def load_all_gps(test_evid, stas, keys):
    all_gps = dict()
    for sta in stas:
        all_gps[sta] = dict()
        print "loading", sta
        for key in keys:
            #print "loading", sta, key
            fname = os.path.join(basedir, str(test_evid), 'gps', sta, key + '.gp')
            all_gps[sta][key] = GP(fname=fname, build_tree=True)
    return all_gps

def run_real_data(stas, doublet_evid, test_evid):
    # load signals
    evs = dict()
    signals = dict()
    for sta in stas:
        evs[sta], signals[sta] = load_vdec_signals(sta)

    # get pick times
    ys = load_true_data(stas)

    # load aligned atimes and extract wavelets
    ys, pairs = do_align_atimes(stas, evs, signals, ys)

    wave_keys = ['w%03d' % i for i in range(151)]
    for sta in stas:
        wk = extract_wavelets_sta(ys[sta], signals[sta], ys[sta]['arrival_time_inferred'], wave_keys=wave_keys)

    train_evs, train_ys, test_ys, ats_to_load, test_ev = isolate_test_ev(evs, ys, test_evid)

    #print repr(ats_to_load)


    import cPickle
    try:
        fname = os.path.join(basedir, 'test_signals_%d.pkl' % test_evid)
        with open(fname) as f:
            signals_test = cPickle.load(f)
    except IOError as e:
        print "couldn't load %s: %s" % (fname, e)
        print "You might need to load signal on vDEC using these atimes:"
        print repr(ats_to_load)
        return


    keys = [k[3:] for k in sorted(ys[ys.keys()[0]].keys()) if k.startswith('obsw')] + ["tt_residual",]
    try:
        all_gps = load_all_gps(test_evid, stas, keys)
    except IOError as e:
        print e

        all_gps = train_all_gps_opt(train_evs, train_ys, test_evid)
        #save_all_gps(test_evid, all_gps)
        print "stopping now since we probably leaked a ton of memory. run again to reload the saved gps."
        return


    #x0 = (73.333333333333329, 40.0, 1266367151.0847535) # grid search
    #x1 = (74.9, 39.6, 1266367147) # 1000 steps of burnin
    burnin_atime = 1000
    samples_atime = 10000
    burnin_combined = 300
    samples_combined = 2000


    x0 = [test_ev.lon, test_ev.lat, test_ev.time]
    fname = "samples_atime3_%d" % test_evid
    ev_mcmc, samples = run_MH(test_ys, all_gps, signals_test, test_ev, burnin=burnin_atime, steps=samples_atime, x0=x0, waves=False, X_train=train_evs['MKAR'], base_start=200, base_end=800, wtype='db9', width_deg=2.5, visualize=False, sample_fname=fname)
    np.save(fname, samples)
    print "saved samples to", fname

    for width_deg in (0.5, 1.0, 2.5):
        visualize_MCMC_state(test_ev, samples[burnin_atime:,:], test_ys, signals_test, all_gps,
                             train_evs['MKAR'], base_start=200, base_end=800, wtype='db9',
                             waves=True, width_deg=width_deg, alphas="uniform",
                             offmap_station_arrows=True, scale=2.0)
        fname = "atime_posterior3_%d_%.2f.png" % (test_evid , width_deg)
        plt.savefig(fname)
        print "saved to", fname

    return
    x0 = samples[-1, 0:3]

    fname = "samples_combined2_%d" % test_evid
    ev_mcmc, samples = run_MH(test_ys, all_gps, signals_test, test_ev, x0=x0, burnin=burnin_combined, steps=samples_combined, waves=True, X_train=train_evs['MKAR'], base_start=200, base_end=800, wtype='db9', width_deg=2.5, visualize=False, sample_fname=fname)
    np.save(fname, samples)
    print "saved samples to", fname

    for width_deg in (0.5, 1.0, 2.5):
        visualize_MCMC_state(test_ev, samples[burnin_combined:,:], test_ys, signals_test, all_gps,
                             train_evs['MKAR'], base_start=200, base_end=800, wtype='db9',
                             waves=True, width_deg=width_deg, alphas="uniform",
                             offmap_station_arrows=True, scale=2.0)
        fname = "combined_posterior2_%d_%.2f.png" % (test_evid, width_deg)
        plt.savefig(fname)
        print "saved to", fname


def plot_heatmaps():
    # test_ys, all_gps, signals_test, test_ev, x0=x1, burnin=0, steps=1000, waves=True,
    #                            X_train=train_evs['MKAR'], base_start=200, base_end=800, wtype='db9')

    ev_true=test_ev
    ev_tmp = Event(lon=ev_true.lon, lat=ev_true.lat, depth=ev_true.depth, mb=ev_true.mb, time=ev_true.time)
    atimes =  {"AAK": test_ys['AAK']['arrival_time_loaded'] - 15.0/20,
                  "AKTO": test_ys['AKTO']['arrival_time_loaded'] - 2.0/20,
                  "BVAR": test_ys['BVAR']['arrival_time_loaded'] - 36.0/20,
                  "MKAR": test_ys['MKAR']['arrival_time_loaded'] - 4.0/20,
                  "ZALV": test_ys['ZALV']['arrival_time_loaded'] - 14.0/20}

    logscale=True
    hms = dict()
    for sta in signals_test.keys():
        def f(lon, lat):
            ev_tmp.lon=lon
            ev_tmp.lat = lat
            if logscale:
                return joint_log_likelihood_full(ev_tmp, test_ys, atimes, signals_test, all_gps, base_start=200, base_end=800, wtype='db9', only_stas=[sta,], times=False)
            else:
                return np.exp(joint_log_likelihood_full(ev_tmp, test_ys, atimes, signals_test, all_gps, base_start=200, base_end=800, wtype='db9', only_stas=[sta,]), times=False)


        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        n=40
        hm = EventHeatmap(f=f, calc=True, n=n, top_lat=40.0, bottom_lat=39.0, left_lon=74.5, right_lon=75.5, fname="stuff_%s_wavelet4" % sta)
        nans = np.isnan(hm.fvals)
        hm.fvals[nans] = np.min(hm.fvals[~nans])
        #hm.fvals = np.exp(hm.fvals - np.max(hm.fvals))
        hm.add_stations(stas)
        #hm.init_bmap(axes=ax, nofillcontinents=True)
        #hm.plot_earth()
        #hm.plot_density(smooth=True)
        hm.set_true_event(test_ev.lon, test_ev.lat)
        hm.plot(axes=ax, smooth=True)
        hm.plot_locations([(x[0], x[1]) for x in train_evs[sta]], marker='x', ms=4, mec="black", mew=1, alpha=1.0, c='black')
        #hm.bmap.drawmapscale(75.1, 39.6, 75.1, 39.6, 10, fontcolor='white')
        ax.set_title(sta)
        print "plotted", sta
        hms[sta] = hm

    hm_global = None
    for sta in hms.keys():
        if hm_global is None:
            hm_global = hms[sta]
        else:
            hm_global = hm_global + hms[sta]

    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(1,1,1)
    #hm_global.fvals = np.exp(hm_global.fvals - np.max(hm_global.fvals))
    hm_global.add_stations(stas)
    hm_global.set_true_event(test_ev.lon, test_ev.lat)
    hm_global.plot(axes=ax, smooth=True)
    hm_global.plot_locations([(x[0], x[1]) for x in train_evs[sta]], marker='x', ms=4, mec="black", mew=1, alpha=1.0, c='black')
    hm_global.bmap.projection="mill"
    hm_global.bmap.drawmapscale(75.1, 39.6, 75.1, 39.6, 50)
    hm_global.bmap.projection="cyl"
    ax.set_title("evid %d: location posterior log-density from wavelet features alone" % (test_evid))



def main_synthetic():
    X_events = select_events(train=True)
    X_test = select_events(train=False)
    ev = gen_test_ev(X_test[6,:])
    n=15
    wparams = ['w%03d' % i for i in range(n)]
    ys = sample_synthetic_data(stas, wparams, X_events)

    all_gps = train_all_gps(X_events, ys)

    test_ys = sample_test_values(ev, all_gps)
    test_signals = dict([(sta, gen_signals_sta(test_ys[sta]).flatten()) for sta in stas])

    ev_mcmc, samples = run_MH(test_ys, all_gps, test_signals, ev, x0=(22.22, 36.67,1238262509.1), burnin=0, steps=1000, waves=True, X_train=X_events)

def main():
    """
    From ipython:
RESULTS: (all distance from LEB: obviously the real distance are probably more like 1KM)
(7461352, 7462928) are 14.5km apart and match at 5 stations (AAK, BVAR, KURK, SONM, ZALV)
(5606252, 5606134) are 17.64km apart and match at 4 stations (BVAR, KURK, MKAR, ZALV)
(4760683, 5891632) are 4.3km apart and match at 4 stations (AAK, AKTO, BVAR, MKAR)
(5570334, 5570256) are 6.46km apart and match at 4 stations (ZALV, MKAR, KURK, AAK)
"""


    stas = ['AAK', 'AKTO', 'BVAR', 'KURK', 'MKAR', 'ZALV']
    #
    #doublet_evid, test_evid = (5606252, 5606134)
    import sys
    if len(sys.argv) > 2:
        doublet_evid = int(sys.argv[1])
        test_evid = int(sys.argv[2])
    else:
        doublet_evid, test_evid = (4760683, 5891632)

    run_real_data(stas, doublet_evid, test_evid)

if __name__ == "__main__":
    main()
