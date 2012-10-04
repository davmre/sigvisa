import numpy as np

#from database import db, dataset

class NestedDict(dict):
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, NestedDict())

class Sigvisa(object):

    # singleton pattern -- only initialize once
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Sigvisa, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance


    def __init__(self):

        st = 1237680000
        et = st + 24 * 3600
        dbconn = db.connect()
        self.cursor = dbconn.cursor()
        self.sites = dataset.read_sites(cursor)
        self.stations, self.siteids = dataset.read_sites_by_name(cursor)
        self.site_up = dataset.read_uptime(cursor, st, et)
        self.phasenames, self.phasetimedef = dataset.read_phases(cursor)
        self.phaseids = dict(zip(self.phasenames, range(len(self.phasenames))))
        self.earthmodel = learn.load_earth("parameters", self.sites, self.phasenames, self.phasetimedef)
        self.sigmodel = learn.load_sigvisa("parameters", st, et, "spectral_envelope", self.site_up, self.sites, self.phasenames, self.phasetimedef, load_signal_params = False)
        


        self.noise_models = NestedDict()


        self.bands = ("freq_2_3",)
        self.chans = ('BHZ',)
        self.phases = ('P', 'Pn', 'Pg', 'S', 'Sn', 'Lg')

        self.P_phases = ('P', 'Pn')
        self.S_phases = ('S', 'Sn', 'Lg')





    def phasenames(self, phase_id_minus1_list):
        return [self.phasenames[id] for id in phase_id_minus1_list]




    def get_noise_model(self, waveform=None, sta=None, chan=None, filter_str=None, time=None, order=17):
        """
        Returns an ARModel noise model of the specified order for the
        given station/channel/filter, trained from the hour prior to
        the given time. Results are cached, so each noise model is
        learned only once.

        This method takes *either* a station, channel, filter string,
        and time, *or* a Waveform object from which these can all be
        read.

        """

        if waveform is not None:
            sta = waveform['sta']
            chan = waveform['chan']
            filter_str = waveform['filter_str']
            time = waveform['stime']
        else:
            if sta is None or chan is None or filter_str is None or time is None:
                raise Exception("missing argument to get_noise_model!")
        

        hour = int(time/3600)

        # check to see if we have this model in the cache
        armodel = self.noise_models[sta][chan][filter_str][order][hour]
        if isinstance(armodel, ARModel):
            return armodel

        noise_hour_start = (hour-1)*3600
        noise_hour_end = hour*3600

        arrivals = read_station_detections(sta, chan, noise_hour_start, noise_hour_end)
        arrival_times = arrivals[:, DET_TIME_COL]

        # load waveforms for five minutes within the previous hour 
        waves = []
        while len(waves) < 5:
            minute = np.random.randint(60)*60+noise_hour_start

            # skip any minute that includes a detected arrival
            for t in arrival_times:
                if t > minute and t < minute+60:
                    continue

            wave = fetch_waveform(sta, chan, minute, minute+60)

            # also skip any minute for which we don't have much data
            if wave['fraction_valid'] < 0.5:
                continue

            waves.append(wave)

        # choose the median minute as our model
        waves.sort(key=lambda w : np.mean(w.data))
        model_wave = waves[2][filter_str]

        # train AR noise model
        ar_learner = ARLearner(model_wave.data, model_wave['srate'])
        params, std = ar_learner.yulewalker(order)
        em = ErrorModel(0, std)
        armodel = ARModel(params, em, c = ar_learner.c)
        self.noise_models[sta][chan][filter_str][order][hour] = armodel
        return armodel
    
        
    def arriving_phases(self, event, sta):

        siteid = self.siteids[sta]
        phases = [p for p in self.phases where self.sigvisa.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid-1, self.phaseids[phase]-1) > 0]
        return phases


                
