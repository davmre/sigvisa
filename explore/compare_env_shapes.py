import numpy as np
import scipy.optimize
import itertools
import pandas as pd

from sigvisa import Sigvisa
from sigvisa.signals.io import load_event_station_chan


grid = dict()
grid['exp'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (-1.0, 1.0), (-5, -3, -1))])
grid['poly'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5))])
grid['polyexp'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5), (-7, -3, -1))])
grid['expexp'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-2.0, -0.1),  (-4, -2), (-7, -3))])

grid_plateau = dict()
grid_plateau['exp'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-5, -3, -1))])
grid_plateau['poly'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5))])
grid_plateau['polyexp'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5), (-7, -3, -1))])
grid_plateau['expexp'] = np.array([x for x in itertools.product( (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (-2.0, -0.1),  (-4, -2), (-7, -3))])

grid_noise = dict()
grid_noise['exp'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (-1.0, 1.0), (-5, -3, -1))])
grid_noise['poly'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5))])
grid_noise['polyexp'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5), (-7, -3, -1))])
grid_noise['expexp'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-2.0, -0.1),  (-4, -2), (-7, -3))])

grid_plateau_noise = dict()
grid_plateau_noise['exp'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-5, -3, -1))])
grid_plateau_noise['poly'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5))])
grid_plateau_noise['polyexp'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-2, -.5, .5), (-7, -3, -1))])
grid_plateau_noise['expexp'] = np.array([x for x in itertools.product( (-3.0, -0.5), (26, 30, 32 ), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (-2.0, -0.1),  (-4, -2), (-7, -3))])


def decay_exp(t, amp, decay):
    return amp * np.exp(-t * np.exp(decay))

def decay_poly(t, amp, decay):
    return amp * (t+1)**-np.exp(decay)

def decay_polyexp(t, amp, decay_poly, decay_exp):
    return amp * (t+1)**-np.exp(decay_poly) * np.exp(- t * np.exp(decay_exp))

def decay_expexp(t, peak_amp, amp2, decay1, decay2):
    return decay_exp(t, (peak_amp-amp2), decay1) + decay_exp(t, amp2, decay2)

def env_plateau(params, n, srate, decay_fn):
    amp_noise = np.exp(params[0])
    t_start = params[1]
    t_rise = np.exp(params[2])
    t_plateau = np.exp(params[3])
    amp = np.exp(params[4])

    signal = np.zeros((n,))
    n_start = int(t_start * srate)
    n_rise = int((t_start + t_rise) * srate)
    signal[n_start: n_rise] = np.linspace(0, amp, n_rise-n_start)
    n_plateau = int( (t_start + t_rise + t_plateau) * srate)
    signal[n_rise:n_plateau] = amp

    t = np.linspace(0, float(n-n_plateau+1)/srate, n-n_plateau+1)
    signal[n_plateau:] = decay_fn(t[1:], amp, *params[5:])
    signal += amp_noise
    return signal

def loss(params, wave, decay_fn, plateau, noise):
    n = wave['npts']
    srate = wave['srate']

    try:
        if not noise:
            params = np.concatenate(((-999,), params))
        if not plateau:
            params = np.concatenate((params[:3], (-999,), params[3:]))
        env = env_plateau(params, n, srate, decay_fn)

    except:
        #raise
        return 1e10


    diff = wave.data - env

    l1 = np.sum(np.abs(diff))
    return l1




def fit_model(cursor, evid, sta, model_type="exp", plateau=False, noise=False):
    decay_fn = eval("decay_"+ model_type)

    try:
        wave = load_event_station_chan(evid, sta, 'auto', cursor=cursor).filter("%s;hz_10;env" % 'freq_2.0_3.0')
    except:
        return np.float('nan'), []

    if noise:
        mygrid = grid_noise[model_type] if not plateau else grid_plateau_noise[model_type]
    else:
        mygrid = grid[model_type] if not plateau else grid_plateau[model_type]
    gridvals = np.zeros(mygrid.shape[0])
    for (j, gval) in enumerate(mygrid):
        gridvals[j] = loss(gval, wave, decay_fn, plateau, noise)

    optimize_best = 5
    best_results = np.zeros((optimize_best,))
    best_indices = np.zeros((optimize_best,), dtype=int)
    best_gvals = np.zeros((optimize_best,mygrid.shape[1]))
    for i in range(optimize_best):
        best_indices[i] = np.argmin(gridvals)
        gridvals[best_indices[i]] = np.float('inf')

        r = scipy.optimize.minimize(loss, x0 = mygrid[best_indices[i],:], args=(wave, decay_fn, plateau, noise), method='Nelder-Mead')
        best_results[i] = r.fun
        best_gvals[i,:] = r.x
    ii = np.argmin(best_results)
    return best_results[ii], best_gvals[ii,:]

def main():
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    with open('evid_list') as f:
        lines = f.readlines()

    p = np.random.permutation(len(lines)-1)[:10000]

    pairs = [lines[pi].strip().split() for pi in p]
    pairs = [(sta, int(evid)) for (sta, evid) in pairs]
    results = pd.DataFrame(data = pairs, columns = ['sta', 'evid'])
    cursor.close()


    cursor = s.dbconn.cursor()

    for model_type in ('exp', 'poly', 'polyexp', 'expexp'):
        for plateau in (False, True):
            for noise in (False, True):
                col_label = model_type if not plateau else model_type + "_plateau"
                col_label = col_label if not noise else col_label + "_noise"
                results[col_label] = np.zeros((results.shape[0],))


    for (i, srs) in results.iterrows():
        for model_type in ('exp', 'poly', 'polyexp', 'expexp'):
            for plateau in (False, True):
                for noise in (False, True):
                    col_label = model_type if not plateau else model_type + "_plateau"
                    col_label = col_label if not noise else col_label + "_noise"

                    evid = srs['evid']
                    sta = srs['sta']
                    ll, params = fit_model(cursor, evid, sta, model_type=model_type, plateau=plateau, noise=noise)
                    results[col_label][i] = ll

        if i % 100 == 0:
            fname = 'out_%d.csv' % i
            print "saving partial results to", fname
            results.to_csv(fname)

    results.to_csv('out_final.csv')
    cursor.close()

if __name__ == "__main__":
    main()
