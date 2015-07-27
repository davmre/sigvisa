import numpy as np
import cPickle as pickle
import os
from sigvisa.database.signal_data import get_fitting_runid
from sigvisa.learn.train_param_common import insert_model
from sigvisa.source.event import Event
from sigvisa.utils.geog import dist_km
from sigvisa.utils.fileutils import mkdir_p
from sigvisa.models.wiggles.wavelets import construct_full_basis_implicit
from sigvisa.treegp.gp import GPCov, prior_sample
from sigvisa.models.spatial_regression.SparseGP import SparseGP
from sigvisa import Sigvisa
from sigvisa.graph.sigvisa_graph import SigvisaGraph
from sigvisa.signals.common import Waveform
from sigvisa.models.ttime import tt_predict
from sigvisa.models.distributions import Uniform, Poisson, Gaussian, Exponential, TruncatedGaussian
from sigvisa.synthetic import sample_event
import copy


class SampledWorld(object):

    def __init__(self, seed=0):
        self.seed = seed
        pass

    def sample_sg(self, *args, **kwargs):
        evs, waves, sg = sample_event(*args, seed=self.seed,return_all=True,  **kwargs)
        self.evs = evs
        self.all_evs = evs
        self.n_evs = len(evs)
        self.locs = [(ev.lon, ev.lat) for ev in evs]
        self.ev_doublet = None
        self.ev_doublet_base = None

        w = dict()
        w[-1] = dict()
        for wave in waves:
            w[-1][wave['sta']] = wave
        self.waves = w

    def sample_region(self, n_evs, lons, lats, times, mbs, doublet_idx=None, doublet_dist=0.1):

        locs = []
        evs = []
        np.random.seed(self.seed)
        for n in range(n_evs):
            lon = np.random.rand() * (lons[1]-lons[0]) + lons[0]
            lat = np.random.rand() * (lats[1]-lats[0]) + lats[0]
            time = np.random.rand() * (times[1]-times[0]) + times[0]
            mb = np.random.rand() * (mbs[1]-mbs[0]) + mbs[0]
            locs.append((lon, lat))
            ev = Event(lon=lon, lat=lat, time=time, depth=0, mb=mb, eid=n+1)
            evs.append(ev)
        locs = np.array(locs)

        if doublet_idx is not None:
            ev_doublet_base = evs[doublet_idx]
            mb = np.random.rand() * (mbs[1]-mbs[0]) + mbs[0]
            ev_doublet = Event(lon=ev_doublet_base.lon + (np.random.rand()-.5)*doublet_dist, lat=ev_doublet_base.lat+ (np.random.rand()-.5)*doublet_dist, time= np.random.rand() * (times[1]-times[0]) + times[0],  mb=mb, depth=0, eid=n_evs+1)
            doublet_dist = dist_km((ev_doublet_base.lon, ev_doublet_base.lat), (ev_doublet.lon, ev_doublet.lat))
            self.ev_doublet = ev_doublet
            self.ev_doublet_base = ev_doublet_base

        self.n_evs = n_evs
        self.evs = evs
        self.locs = locs
        self.all_evs = evs  + [ev_doublet,]

    def add_events(self, evs):
        locs = []
        for ev in evs:
            locs.append((ev.lon, ev.lat))
        self.locs = locs
        self.evs = evs
        self.all_evs = evs
        self.n_evs = len(evs)
        self.ev_doublet = None
        self.ev_doublet_base = None

    def load_param_models(self, runids, tmtypes=None, phase="P"):

        template_shape="lin_polyexp"
        tmtypes = {'tt_residual': 'constant_laplacian',
                   'peak_offset': 'param_linear_mb',
                   'mult_wiggle_std': 'dummy',
                   'coda_decay': 'param_linear_distmb',
                   'peak_decay': 'param_linear_distmb',
                   'amp_transfer': 'param_sin1'} \
            if tmtypes is None else tmtypes

        from sigvisa.graph.sigvisa_graph import get_param_model_id
        from sigvisa.learn.train_param_common import load_modelid
        pmodels = dict()
        for sta in self.stas:
            pmodels[sta] = dict()
            for param in ("tt_residual", "amp_transfer", "coda_decay", "peak_decay", "peak_offset"):
                modelid = get_param_model_id(runids, sta, phase, tmtypes[param], param, template_shape, chan=None, band=None)
                model = load_modelid(modelid)
                pmodels[sta][param] = model
            pmodels[sta]['mult_wiggle_std'] = Gaussian(0.6, 0.1)

        return pmodels

    def joint_sample_arrival_params(self, gpcov, param_means, coef_noise_var=0.01, param_noise_var=0.1, param_models=None, params_as_gp_base=None):
        """
        for each station, sample signals jointly for all events.

        this means:
        - get a GP for each wavelet coef (since I'm god it can be all the same GP...)
        - for each wavelet coef, sample values jointly for all stations
        -
        """

        evs, ev_doublet = self.evs, self.ev_doublet
        n_evs = self.n_evs
        n_coefs = self.n_coefs
        self.coef_noise_var = coef_noise_var
        self.param_noise_var = param_noise_var
        self.param_means = param_means
        self.gpcov = gpcov


        X = np.array([(ev.lon, ev.lat, ev.depth, 0.0, ev.mb) for ev in self.all_evs], dtype=float)
        self.X = X

        np.random.seed(self.seed)
        true_coefs = dict()
        tm_params = dict()
        y, K = prior_sample(X, gpcov, coef_noise_var, return_K=True)
        for sta in self.stas:
            tm_params[sta] = dict()
            for param in param_means[sta].keys():
                if param_models is not None and param in param_models[sta]:
                    tm_params[sta][param] = np.array([float(param_models[sta][param].sample(X[i:i+1,:])) for i in range(n_evs+1)])
                else:
                    tm_params[sta][param]  = prior_sample(X, gpcov, param_noise_var) + param_means[sta][param]

            true_coefs[sta] = np.zeros((len(self.all_evs), n_coefs))
            for i in range(n_coefs):
                coefs = prior_sample(X, gpcov, coef_noise_var)
                true_coefs[sta][:, i] = coefs
            print "sampled true coefs at", sta

        self.tm_params = tm_params
        self.true_coefs = true_coefs
        return tm_params, true_coefs

    def sample_signals(self, band, phase="P"):
        """
        Given coefficients, sample signals for each event/station.
        """
        evs, ev_doublet, wavelet_family, stas = self.evs, self.ev_doublet, self.wavelet_family, self.stas
        tm_params, true_coefs, basis = self.tm_params, self.true_coefs, self.basis
        self.band = band
        self.phase=phase
        self.chans = dict()

        s = Sigvisa()
        waves = dict()
        np.random.seed(self.seed)
        for i, ev in enumerate(self.all_evs):
            sg = SigvisaGraph(template_model_type="dummy", template_shape="lin_polyexp",
                                  wiggle_model_type="dummy", wiggle_family=wavelet_family,
                                  nm_type = "ar", phases=[phase,], runids=(), )
            wns = dict()
            for sta in stas:
                stime = ev.time + tt_predict(ev, sta, phase) - 100
                chan = s.canonical_channel_name[s.default_vertical_channel[sta]]
                self.chans[sta] = chan
                wave = Waveform(data = np.zeros(2000), srate=self.srate, stime=stime, sta=sta, chan=chan, filter_str="%s;env;hz_%.1f" % (band, self.srate))
                wns[sta] = sg.add_wave(wave)

            sg.add_event(ev)

            waves[i]=dict()
            for (sta, wn) in wns.items():

                tmnodes = sg.get_template_nodes(1, sta, phase, wn.band, wn.chan)
                for p, (k, n) in tmnodes.items():
                    if p in tm_params[sta]:
                        n.set_value(tm_params[sta][p][i])

                wn.wavelet_basis = basis
                wn.wavelet_param_models[phase] = [Gaussian(c, 1e-8) for c in true_coefs[sta][i,:]]

                wn.unfix_value()
                wn.parent_sample()
                waves[i][sta] = wn.get_wave()

        self.waves = waves
        return waves

    def set_basis(self, wavelet_family, srate=5.0):
        self.srate=srate
        self.wavelet_family=wavelet_family
        if wavelet_family.startswith("db"):
            self.basis = construct_full_basis_implicit(srate=srate,
                                                       wavelet_str=wavelet_family,
                                                       c_format=True)
            self.n_coefs = len(self.basis[0])
        else:
            self.n_coefs = 0

    def serialize(self, wave_dir):
        """
        dump sampled waveforms...
        """
        mkdir_p(wave_dir)

        waves = self.waves
        for i in waves.keys():
            for sta in waves[i].keys():
                wave = waves[i][sta]
                f = open(os.path.join(wave_dir, "wave_%s_%d" % (sta, i)), 'wb')
                pickle.dump(wave, f)
                f.close()

        f = open(os.path.join(wave_dir, "events.txt"), 'w')
        for ev in self.evs:
            f.write("%d\t%f\t%f\t%f\t%f\t%f\t%d\n" % (ev.eid, ev.lon, ev.lat, ev.depth,
                                                      ev.time, ev.mb, ev.natural_source))
        f.close()

        with open(os.path.join(wave_dir, "events.pkl"), 'wb') as f:
            pickle.dump(self.evs, f)

        for sta in self.true_coefs.keys():
            np.savetxt(os.path.join(wave_dir, "coefs_%s.txt" % sta), self.true_coefs[sta])

        with open(os.path.join(wave_dir, "params.txt"), 'w') as f:
            f.write(repr(self.tm_params))

        with open(os.path.join(wave_dir, "sampled_world.pkl"), 'wb') as f:
            pickle.dump(self, f)

    def save_gps(self, wave_dir, run_name):
        s = Sigvisa()
        cursor = s.dbconn.cursor()
        runid = get_fitting_runid(cursor, run_name, 1, create_if_new = True)
        model_dir = os.path.join(wave_dir, run_name)
        mkdir_p(model_dir)
        for sta in self.train_gps.keys():
            for param in self.train_gps[sta].keys():
                model = self.train_gps[sta][param]
                param_str = param
                if type(param_str)==int:
                    param_str = self.wavelet_family + "_%d" % param

                model_fname = os.path.join(model_dir, "%s_%s.gp" % (sta, param_str))
                model.save_trained_model(model_fname)
                modelid =  insert_model(s.dbconn, runid, param_str, sta, self.chans[sta], self.band, self.phase, "gp_lld", model_fname, training_set_fname="", training_ll=model.log_likelihood(), require_human_approved=False, max_acost=1e9, n_evids=self.n_evs, min_amp=0.0, elapsed=0.0, template_shape="lin_polyexp" )

    def train_gp_models_true_data(self):
        tm_params = self.tm_params
        true_coefs = self.true_coefs
        gpcov = self.gpcov
        n_evs = self.n_evs
        param_means, param_noise_var, coef_noise_var = self.param_means, self.param_noise_var, self.coef_noise_var

        trainX = self.X[:n_evs, :].copy()
        train_gps = dict()
        for sta in self.stas:
            train_gps[sta] = dict()
            for param in tm_params[sta].keys():
                y = tm_params[sta][param][:n_evs]
                mean = param_means[sta][param]
                gp = SparseGP(trainX, y, noise_var=param_noise_var, cov_main=gpcov,
                              ymean=mean, sta=sta, compute_ll=True)
                train_gps[sta][param] = gp

            for i in range(true_coefs[sta].shape[1]):
                y = true_coefs[sta][:n_evs, i]
                gp = SparseGP(trainX, y, noise_var=coef_noise_var, cov_main=gpcov,
                              sta=sta, compute_ll=True)
                train_gps[sta][i] = gp
        self.train_gps = train_gps


def load_sampled_world(wave_dir):
    with open(os.path.join(wave_dir, "sampled_world.pkl"), 'rb') as f:
        sw = pickle.load(f)
    return sw



def build_param_means(stas, tt_residual=0.0, coda_decay=-3.0, peak_decay=-3.0, peak_offset=0.0, amp_transfer=3.0, mult_wiggle_std=0.6):
    param_means = dict()
    param_mean_base = {"tt_residual": tt_residual,
                       "coda_decay": coda_decay,
                       "peak_decay": peak_decay,
                       "peak_offset": peak_offset,
                       "mult_wiggle_std": mult_wiggle_std,
                       "amp_transfer": amp_transfer}
    for sta in stas:
        param_means[sta] = copy.copy(param_mean_base)
    return param_means

def sample_params_and_signals(basedir, seed=0):
    n_evs = 10
    lons = [129, 130]
    lats = [-3.5, -4.5]
    times = [1240240000, 1240340000]
    mbs = [4.0, 5.0]
    sw = SampledWorld(seed=seed)
    sw.sample_region(n_evs, lons, lats, times, mbs, doublet_idx=1)
    sw.stas = ["MK31", "AS12", "CM16", "FITZ", "WR1"]
    gpcov = GPCov([0.7,], [ 40.0, 5.0],
                  dfn_str="lld",
                  wfn_str="compact2")
    param_means = build_param_means(sw.stas)
    sw.set_basis(wavelet_family="db4uvars_2.0_3_30_0.4", srate=5.0)
    sw.joint_sample_arrival_params(gpcov, param_means)
    sw.sample_signals("freq_0.8_4.5")

    wave_dir = os.path.join(basedir, "sampled_%d" % seed)
    sw.serialize(wave_dir)
    sw.train_gp_models_true_data()
    sw.save_gps(wave_dir, run_name="synth_truedata")


def build_doublet_sg(sw, runid, ev_init=None, **kwargs):
    ev_doublet = sw.ev_doublet

    kwargs['wiggle_family'] = sw.wavelet_family if "wiggle_family" not in kwargs else kwargs['wiggle_family']

    ev_doublet_init = Event(lon=ev_doublet.lon+1, lat=ev_doublet.lat-1,
                            depth=ev_doublet.depth, mb=ev_doublet.mb+0.4,
                            time=ev_doublet.time+20.0)

    sg = SigvisaGraph(template_shape="lin_polyexp",
                              nm_type = "ar", phases=["P"], runids=(runid,), **kwargs)
    wns = dict()
    for sta in sw.stas:
        wns[sta] = sg.add_wave(sw.waves[sw.n_evs][sta])
        wns[sta].wavelet_basis = sw.basis, sw.gpcov.wfn_params[0]

    evnodes = sg.add_event(ev_doublet_init if ev_init is None else ev_init)
    return sg, wns

def build_joint_sg(sw, runid, init_evs="true", init_templates=True, **kwargs):
    ev_doublet = sw.ev_doublet

    kwargs['wiggle_family'] = sw.wavelet_family if "wiggle_family" not in kwargs else kwargs['wiggle_family']

    sg = SigvisaGraph(template_shape="lin_polyexp",
                              nm_type = "ar", phases=["P"], runids=(runid,), **kwargs)
    wns = dict()
    for w in sw.waves.values():
        for sta in w.keys():
            wns[sta] = sg.add_wave(w[sta])
            #wns[sta].wavelet_basis = sw.basis

    for ev in sw.all_evs:
        if init_evs=="none": continue
        elif init_evs.startswith("noise"):
            ev_init = Event(lon=ev.lon+1*np.random.randn(),
                            lat=ev.lat-1*np.random.randn(),
                            depth=ev.depth,
                            mb=ev.mb+0.4 *np.random.randn(),
                            time=ev.time+20.0*np.random.randn())
        elif init_evs=="true":
            ev_init = ev
        evnodes = sg.add_event(ev_init)

    if init_templates:
        assert(init_evs != "none")
        for sta in sw.tm_params.keys():
            for eid in range(1, len(sw.all_evs)+1):
                tmnodes = sg.get_template_nodes(eid, sta, sw.phase, chan=sw.chans[sta], band=sw.band)
                for param in sw.tm_params[sta].keys():
                    k, n = tmnodes[param]
                    n.set_value(sw.tm_params[sta][param][eid-1])

    return sg, wns


def save_hparams(sg, mcmc_dir, vals=None):
    if vals is None:
        vals = {'noise_var': 0.01,
                'signal_var': 1.0,
                'horiz_lscale': 300,
                'depth_lscale': 5.0}

    true_params = dict()
    for sta in sg.station_waves.keys():
        true_params[sta] = dict()
        nparams = len(sta.wavelet_basis[0][0])
        for hparam in sg.jointgp_hparam_prior.keys():
            true_params[sta][hparam] = [vals[hparam],]*nparams
    with open(os.path.join(mcmc_dir, 'gp_hparams', 'true.pkl'), 'wb') as f:
        pickle.dump(true_params, f)

def main():
    basedir = os.path.join(os.getenv("SIGVISA_HOME"), "experiments", "synth_wavematch")
    wave_dir = os.path.join(basedir, "sampled_%d" % 0)
    sw = load_sampled_world(wave_dir)
    sg, wns = build_sg(sw, 34)

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import sys, traceback, pdb
        print e
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
