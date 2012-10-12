

class TemplateModel(object):
    """
    Abstract class defining a signal template model.
    
    A phase template is defined by some number of parameters
    (e.g. onset period, height, and decay rate for a
    paired-exponential template). A signal consists of a number of
    phase arrivals. 

    The methods in the class deal with matrices, each row of which
    gives the parameters for a specific phase arrival. That is, we
    allow modeling the joint distribution of template parameters over
    multiple phases, though it's also possible for a particular
    implementation to treat them independently.
    
    Currently we assume that each channel and frequency band are
    independent. This should probably change.

    """

    def params(self):
        raise Exception("method not implemented")

    def predictTemplate(self, event, sta, chan, band, phases=None):
        raise Exception("method not implemented")

    def likelihood(template, event, sta, chan, band, phases=None):
        raise Exception("method not implemented")

    def sample(template, event, sta, chan, band, phases=None):
        raise Exception("method not implemented")

    def travel_time(self, event, sta, phase):
        siteid = self.sigvisa.siteids[sta]
        phaseid = self.sigvisa.phaseids[phase]
        meantt = self.sigvisa.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid-1, phaseid-1)
        return meantt
        
    def sample_travel_time(self, event, sta, phase):
        meantt = self.mean_travel_time(event, sta, phase)

        # peak of a laplace distribution is 1/2b, where b is the
        # scale param, so (HACK ALERT) we can recover b by
        # evaluating the density at the peak
        siteid = sigvisa.siteids[sta]
        phaseid = sigvisa.phaseids[phase]
        ttscale = 2.0 / np.exp(self.sigvisa.sigmodel.arrtime_logprob(0, 0, 0, siteid-1, phaseid-1))

        # sample from a Laplace distribution:
        U = np.random.random() - .5
        tt = meantt - ttscale * np.sign(U) * np.log(1 - 2*np.abs(U))
        return tt
        
    def travel_time_log_likelihood(self, tt, event, sta, phase):
        meantt = self.mean_travel_time(event, sta, phase)
        siteid = sigvisa.siteids[sta]
        phaseid = sigvisa.phaseids[phase]
        ll = self.sigvisa.arrtime_logprob(tt, meantt, 0, siteid-1, phaseid-1)
        return ll



class ExponentialTemplateModel(TemplateModel):

#    target_fns = {"decay": lambda r : r[FIT_CODA_DECAY], "onset": lambda r : r[FIT_PEAK_DELAY], "amp": lambda r: r[FIT_CODA_HEIGHT] - r[FIT_MB], "amp_transfer": lambda r : r[FIT_CODA_HEIGHT] - SourceSpectrumModel().source_logamp(r[FIT_MB], int(r[FIT_PHASEID]), bandid=int(r[FIT_BANDID]))}

    __params =  ("arrival_time", "onset_period", "amp_transfer", "decay")

    def __init__(self, run_name, model_type = "gp_dad"):

        self.sigvisa = Sigvisa()

        # load gp models
        if model_type[0:3] = "gp_":
            self.models = NestedDict()

            for param in self.params():
                basedir = os.path.join("parameters", model_type, param)
                for sta in os.listdir(basedir):
                    sta_dir = os.path.join(basedir, sta)
                    if not os.isdir(sta_dir):
                        continue
                    for phase in os.listdir(sta_dir):
                        phase_dir = os.path.join(sta_dir, phase)
                        if not os.isdir(phase_dir):
                            continue
                        for chan in os.listdir(phase_dir):
                            chan_dir = os.path.join(phase_dir, chan)
                            if not os.isdir(chan_dir):
                                continue
                            for band_model in os.listdir(chan_dir):
                                band_file = os.path.join(chan_dir, band_model)
                                band_run, ext = os.path.splitext(band_model)
                                band, run = os.path.splitext(band_run)

                                if run == run_name:
                                    self.models[param][sta][phase][chan][band] = SpatialGP(fname=band_file)

    def params(self):
        return self.__params

    def predictTemplate(self, event, sta, chan, band, phases=None):
        if phases is None:
            phases = Sigvisa().phases
        
        predictions = np.zeros((len(phases),  len(__params)))
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(params):
                model = self.models[param][sta][phase][chan][band]
                if isinstance(model, NestedDict):
                    raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                if param == "amp_transfer":
                    source_logamp = event.source_logamp(band)
                    predictions[i,j] = source_logamp + model.predict(event)
                elif param == "arrival_time":
                    predictions[i,j] = event.time + self.travel_time(event, sta)
                else:
                    predictions[i,j] = model.predict(event)
        return predictions

    def sample(self, event, sta, chan, band, phases=None):
        if phases is None:
            phases = Sigvisa().phases
        
        samples = np.zeros((len(phases),  len(__params)))
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(params):
                model = self.models[param][sta][phase][chan][band]
                if isinstance(model, NestedDict):
                    raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                if param == "amp_transfer":
                    source_logamp = event.source_logamp(band)
                    samples[i,j] =  source_logamp + model.predict(event)
                elif param = "arrival_time":
                    samples = event.time + self.sample_travel_time(event, sta)
                else:
                    samples[i,j] = model.sample(event)
        return predictions

    def log_likelihood(self, template_params, event, sta, chan, band, phases=None):

        if phases is None:
            phases = Sigvisa().phases
        
        log_likelihood = 0
        for (i, phase) in enumerate(phases):
            for (j, param) in enumerate(params):
                model = self.models[param][sta][phase][chan][band]
                if isinstance(model, NestedDict):
                    raise Exception ("no model loaded for param %s, phase %s (sta=%s, chan=%s, band=%s)" % (param, phase, sta, chan, band))

                if param == "amp_transfer":
                    source_logamp = event.source_logamp(band)
                    log_likelihood += model.log_likelihood(event, template_params[i,j] - source_logamp)
                elif param == "arrival_time":
                    log_likelihood = self.travel_time_log_likelihood(event, sta, template_params[i,j])
                else:
                    log_likelihood += model.log_likelihood(event, template_params[i,j])

        return log_likelihood




def train_param_models(siteids, runids, evid):

    cursor, sigmodel, earthmodel, sites, dbconn = sigvisa_util.init_sigmodel()

    dad_params = {"decay": [.0235, .0158, 4.677, 0.005546, 0.00072], "onset": [1.87, 4.92, 2.233, 0., 0.0001], "amp_transfer": [1.1019, 3.017, 9.18, 0.00002589, 0.403]}


    lldda_sum_params = {"decay": [.01, .05, 1, 0.00001, 20, 0.000001, 1, .05, 300], "onset": [2, 5, 1, 0.00001, 20, 0.000001, 1, 5, 300], "amp": [.4, 0.00001, 1, 0.00001, 20, 0.00001, 1, .4, 800] , "amp_transfer": [.4, 0.00001, 1, 0.00001, 20, 0.00001, 1, .4, 800] }

    gp_params = load_gp_params("parameters/gp_hyperparams.txt", "dad")

    model_dict = NestedDict()

    for siteid in siteids:
        for band in bands:
            for chan in chans:
                print "loading/training siteid %d band %s chan %s" % (siteid, band, chan)
                for (is_s, PSids) in enumerate((P_PHASEIDS, S_PHASEIDS)):

                    short_band = band[16:]
                    fit_data = load_shape_data(cursor, chan=chan, short_band=short_band, siteid=siteid, runids=runids, phaseids=PSids, exclude_evids = [evid,])

                    loaded_dad_params = gp_params[siteid]["S" if is_s else "P"][chan][short_band]
                    if len(loaded_dad_params.keys()) == 3:
                        my_dad_params = loaded_dad_params
                    else:
                        my_dad_params = dad_params

                    try:
                        cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=PSids, chan=chan, target_str="decay", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, dad_params=my_dad_params["decay"], debug=False)
                        model_dict[siteid][band][chan][is_s]["decay"] = cm

                        cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=PSids, chan=chan, target_str="onset", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, dad_params=my_dad_params["onset"], debug=False)
                        model_dict[siteid][band][chan][is_s]["onset"] = cm

                        cm = CodaModel(fit_data=fit_data, band_dir = None, phaseids=PSids, chan=chan, target_str="amp_transfer", earthmodel=earthmodel, sigmodel = sigmodel, sites=sites, dad_params=my_dad_params["amp_transfer"], debug=False)
                        model_dict[siteid][band][chan][is_s]["amp_transfer"] = cm

                    except:
                        import traceback, pdb
                        traceback.print_exc()
                        raise

    return model_dict, cursor, sigmodel, earthmodel, sites
