import numpy as np
import learn.do_training
from database import db, dataset



class NestedDict(dict):
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, NestedDict())

class Sigvisa(object):

    # some channels are referred to by multiple names, i.e. the
    # north-south channel can be BHN or BH2.  here we define a
    # canonical name for each channel. the sigvisa.equivalent_channels(chan)
    # method (defined below) returns a list of all channel names
    # equivalent to a particular channel.
    canonical_channel_name = {"BHZ": "BHZ", "BHN": "BHN", "BHE": "BHE", "BH1": "BHE", "BH2":"BHN"}

    # defined only for canonical channels
    __equivalent_channels = {"BHZ" : ["BHZ"], "BHE": ["BHE", "BH1"], "BHN": ["BHN", "BH2"]}


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
        self.sites = dataset.read_sites(self.cursor)
        self.stations, self.name_to_siteid_minus1, self.siteid_minus1_to_name = dataset.read_sites_by_name(self.cursor)
        self.site_up = dataset.read_uptime(self.cursor, st, et)
        self.phasenames, self.phasetimedef = dataset.read_phases(self.cursor)
        self.phaseids = dict(zip(self.phasenames, range(len(self.phasenames))))
        self.earthmodel = learn.do_training.load_earth("parameters", self.sites, self.phasenames, self.phasetimedef)
        self.sigmodel = learn.do_training.load_sigvisa("parameters", st, et, "spectral_envelope", self.site_up, self.sites, self.phasenames, self.phasetimedef, load_signal_params = False)



        self.noise_models = NestedDict()


        self.bands = ("freq_2.0_3.0",)
        self.chans = ('BHZ', 'BHN', 'BHE')
        self.phases = ('P', 'Pn', 'Pg', 'S', 'Sn', 'Lg')

        self.P_phases = ('P', 'Pn')
        self.S_phases = ('S', 'Sn', 'Lg')

        self.events = dict()

    def phasenames(self, phase_id_minus1_list):
        return [self.phasenames[id] for id in phase_id_minus1_list]






    def arriving_phases(self, event, sta):
        siteid = self.name_to_siteid_minus1[sta] + 1
        phases = [p for p in self.phases if self.sigvisa.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid-1, self.phaseids[phase]-1) > 0 ]
        return phases

    def equivalent_channels(self, chan):
        canonical = self.canonical_channel_name[chan]
        equiv = self.__equivalent_channels[canonical]
        return equiv
