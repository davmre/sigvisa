import matplotlib.pylab as plt
import seaborn as sns
from sigvisa import Sigvisa
import numpy as np

import scipy.stats
import os
from optparse import OptionParser

from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.learn.train_param_common import insert_model

from sigvisa.models.noise.noise_model import NoiseModel
from sigvisa.models.distributions import Gaussian, TruncatedGaussian, InvGamma, MultiGaussian, LogNormal
from sigvisa.utils.fileutils import mkdir_p



def noise_model_model_fname(runid, sta, band, chan, hz, env, n_p, param):
    prefix="parameters/runs/"
    s = Sigvisa()
    fdir = os.path.join(prefix, "noise_%d" % runid, sta, chan, band, "hz_%.2f" % hz)
    mkdir_p(os.path.join(s.homedir, fdir))
    env_s = "env" if env else "raw"
    fname = "%s_np%d_%s.pkl" % (param, n_p, env_s)
    return os.path.join(fdir, fname)

    
def fit_var_model(censored_vars, sta):
    
    # try invgamma
    fit_alpha, fit_loc, fit_beta = scipy.stats.invgamma.fit(censored_vars, floc=0.0)
    ig_model = InvGamma(alpha=fit_alpha, beta=fit_beta)
    ig_lp = np.sum([ig_model.log_p(v) for v in censored_vars])
    
    gmean, gstd = np.mean(censored_vars), np.std(censored_vars)
    norm_model = TruncatedGaussian(gmean, std=gstd, a=0.0)
    norm_lp = np.sum([norm_model.log_p(v) for v in censored_vars])
    
    mu, sigma = np.mean(np.log(censored_vars)), np.std(np.log(censored_vars))
    lognorm_model = LogNormal(mu, sigma)
    lognorm_lp = np.sum([lognorm_model.log_p(v) for v in censored_vars])
        
    if sta=="ELK":
        return lognorm_model, "lognormal", (mu, sigma)
        
        
    try:
        ig_model.variance()
    except:
        print "disallowing invgamma model with undefined variance"
        ig_lp = -np.inf
        
    print ig_lp, norm_lp, lognorm_lp
    if ig_lp > norm_lp and ig_lp > lognorm_lp:
        print "best model is invgamma"
        return ig_model, "invgamma", (fit_alpha,  fit_beta)
    elif norm_lp > lognorm_lp:
        print "best model is tgaussian"
        return norm_model, "tgaussian", (gmean, gstd)
    else:
        print "best model is lognormal"
        return lognorm_model, "lognormal", (mu, sigma)
    
def train_noise_mean_priors(runid, sta, band, hz, env, 
                            mean_upper_cutoff=None, mean_lower_cutoff=None, 
                            std_upper_cutoff = None,
                            std_lower_cutoff = None,
                            chan=None, n_p=None, dummy=False,
                            insert_runid=None, plot=False):

    if insert_runid is None:
        insert_runid = runid
    
    np_cond = ("and nm.n_p=%d" %n_p) if n_p is not None else ""
    fit_conds = "fit.runid=%d and fit.sta='%s' and fit.band='%s' and fit.chan='%s' and fit.hz=%f and fit.env='%s'" %\
                (runid, sta, band, chan, hz, 't' if env else 'f')
    sql_query = "select nm.fname from sigvisa_coda_fit fit, sigvisa_noise_model nm where nm.nmid = fit.nmid and %s %s" % (fit_conds, np_cond)
    s = Sigvisa()
    cursor = s.dbconn.cursor()
    
    cursor.execute(sql_query)
    fnames = cursor.fetchall()
    models = []
    for fname in fnames:
        try:
            model = NoiseModel.load_from_file(fname[0], "ar")
        except:
            continue
        models.append(model)
    
    if len(models) == 0:
        print "WARNING: no models found from query %s, skipping..." % sql_query
        return
    
    means = np.array([nm.c for nm in models])
    stds = np.array([nm.em.std for nm in models])
    params = [np.array(nm.params) for nm in models]
    
    n_ps = [len(p) for p in params]
    r = scipy.stats.mode(n_ps)    
    n_p = r.mode[0]
    params = np.array([ p for p in params if len(p)==n_p])
    
    means_cutoff = np.percentile(means, 90)
    if mean_upper_cutoff is not None:
        means_cutoff = min(mean_upper_cutoff, means_cutoff)
    stds_cutoff = np.percentile(stds, 90)
    if std_upper_cutoff is not None:
        stds_cutoff = min(std_upper_cutoff, stds_cutoff)
        

    censored_means = []
    censored_stds = []
    censored_params = []
    for nm in models:
        if len(nm.params) != n_p: continue
        if nm.c > means_cutoff: continue
        if nm.em.std > stds_cutoff: continue
        if mean_lower_cutoff is not None:
            if nm.c < mean_lower_cutoff: continue
        if std_lower_cutoff is not None:
            if nm.em.std < std_lower_cutoff: continue
            
        censored_means.append(nm.c)
        censored_stds.append(nm.em.std)
        censored_params.append(nm.params)
    censored_means = np.array(censored_means)
    censored_stds = np.array(censored_stds)
    censored_params = np.array(censored_params)
    
    nfits = len(censored_means)
    if nfits < 5:
        print "not enough fits for", sta
        print sql_query
        return

    
        
    fit_loc, fit_scale = scipy.stats.norm.fit(censored_means)
    if env:
        mean_model = TruncatedGaussian(fit_loc, std=fit_scale, a=0)
        mtype="tgaussian"
    else:
        mean_model = Gaussian(fit_loc, std=fit_scale)
        mtype="gaussian"
    if plot:
        plt.figure()    
        sns.distplot(censored_means)
        xs = np.linspace(0, np.max(censored_means), 100)
        meanlps = np.array([np.exp(mean_model.log_p(x)) for x in xs])
        plt.plot(xs, meanlps)
        plt.title("%s means" % sta)
    
    if not dummy:
        phase_name = "noise_%s" % ("env" if env else "raw")
        mean_fname = noise_model_model_fname(insert_runid, sta, band, chan, hz, env, n_p, "mean")
        mean_model.dump_to_file(os.path.join(s.homedir, mean_fname))
        insert_model(s.dbconn, insert_runid, "armean", sta, chan, band, phase_name, 
                     model_type=mtype, model_fname=mean_fname, training_set_fname="", 
                     training_ll=0.0, require_human_approved=False, max_acost=0.0, n_evids=nfits, 
                     min_amp=0.0, elapsed=0.0, hyperparams=repr((fit_loc, fit_scale)))
        print mean_fname

    censored_vars = censored_stds**2    
    var_model, var_model_type, var_hparams = fit_var_model(censored_vars, sta=sta)
    if plot:
        plt.figure()
        sns.distplot(censored_vars, kde=False, norm_hist=True)
        xs = np.linspace(0, np.max(censored_vars), 100)
        varlps = np.array([np.exp(var_model.log_p(x)) for x in xs])
        plt.plot(xs, varlps)
        plt.title("%s vars" % sta)
    
    if not dummy:
        var_fname = noise_model_model_fname(insert_runid, sta, band, chan, hz, env, n_p, "arvar")
        var_model.dump_to_file(os.path.join(s.homedir, var_fname))
        insert_model(s.dbconn, insert_runid, "arvar", sta, chan, band, phase_name, 
                     model_type=var_model_type, model_fname=var_fname, training_set_fname="", 
                     training_ll=0.0, require_human_approved=False, max_acost=0.0, n_evids=nfits, 
                     min_amp=0.0, elapsed=0.0, hyperparams=repr(var_hparams))
        print var_fname
    
    
    mean_params = np.mean(censored_params, axis=0)
    param_cov = np.cov(censored_params.T) + np.eye(n_p) * 0.2
    param_model = MultiGaussian(mean_params, param_cov)
    print "params mean", mean_params
    print "params cov", param_cov
    
    if not dummy:
        param_fname = noise_model_model_fname(insert_runid, sta, band, chan, hz, env, n_p, "arparams")
        param_model.dump_to_file(os.path.join(s.homedir, param_fname))
        insert_model(s.dbconn, insert_runid, "arparams", sta, chan, band, phase_name, 
                     model_type="multigaussian", model_fname=param_fname, training_set_fname="", 
                     training_ll=0.0, require_human_approved=False, max_acost=0.0, n_evids=nfits, 
                     min_amp=0.0, elapsed=0.0, hyperparams=repr((mean_params, param_cov)))
        print param_fname
    
    

def train_envelope_models(runid, sites, band, hz=2, llnl=True):
    for site in sites:
        sta = Sigvisa().get_default_sta(site)
        chan = Sigvisa().default_vertical_channel[sta]
        print sta, chan
        if chan=="sz":
            chan="SHZ"
        if chan=="bz":
            chan="BHZ"

        #if sta=="NEW":
        #    cutoff = 50
        #else:
        #    cutoff = None

        std_cutoff = None
        #if sta=="NV01":
        #    std_cutoff = 100.0
        if sta=="NV01":
            std_cutoff = 100.0
        elif sta=="PFO":
            std_cutoff=20.0
        elif sta=="YBH":
            std_cutoff=10.0
        elif sta=="TX01":
            std_cutoff=20.0
        elif sta=="PD31":
            std_cutoff=100.0
        elif sta=="IL31":
            std_cutoff=25.0
        elif sta=="ANMO":
            std_cutoff=30.0
        elif sta=="IL31":
            std_cutoff=25.0
        elif sta=="ELK":
            std_cutoff=12.0
        cutoff=None
        low_cutoff=None

        train_noise_mean_priors(runid=runid, sta=sta, band=band, chan=chan, hz=hz, 
                                env=True, mean_upper_cutoff=cutoff, std_upper_cutoff=std_cutoff,
                                mean_lower_cutoff=low_cutoff, dummy=False)

def train_raw_models(runid, sites, band, hz=10, llnl=True, insert_runid=None):
    
    if insert_runid is None:
        insert_runid = runid

    hz=10
    llnl=True
    for site in sites:
        sta = Sigvisa().get_default_sta(site)
        chan = Sigvisa().default_vertical_channel[sta]
        print sta, chan
        if chan=="sz":
            chan="SHZ"
        if chan=="bz":
            chan="BHZ"

        std_cutoff=None
        std_low = None
        if sta=="PD31":
            std_cutoff = 120
        elif sta=="YRK8":
            std_cutoff = 20
        elif sta=="ELK":
            std_cutoff = 10
            std_low = 2


        cutoff=None
        low_cutoff=None
        train_noise_mean_priors(runid=runid, sta=sta, band=band, chan=chan, hz=hz, 
                                env=False, mean_upper_cutoff=cutoff, std_upper_cutoff=std_cutoff,
                                std_lower_cutoff=std_low,
                                mean_lower_cutoff=low_cutoff, dummy=False, insert_runid=insert_runid)

def main():


    parser = OptionParser()


    parser.add_option("--band", dest="band", default="freq_0.8_4.5", type="str", help="")
    parser.add_option("--runid", dest="runid", default=None, type="int", help="")
    parser.add_option("--insert_runid", dest="insert_runid", default=None, type="int", help="HACK to record the models under a different runid than the one they were trained on.")
    parser.add_option("--sites", dest="sites", default=None, type="str", help="")
    parser.add_option("--raw", dest="raw", default=False, action="store_true", help="")

    (options, args) = parser.parse_args()

    sites = options.sites
    if sites is None:
        sites="ANMO,ELK,ILAR,KDAK,NEW,NVAR,PDAR,PFO,TXAR,ULM,YBH,YKA"
    sites = sites.split(",")

    if options.raw:
        train_raw_models(options.runid, sites, options.band)
    else:
        train_envelope_models(options.runid, sites, options.band)

if __name__ == "__main__":
    main()
