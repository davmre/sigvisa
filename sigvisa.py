import numpy as np
import learn.do_training
from database import db, dataset
import sigvisa_c

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

        # enforcing the singleton pattern: don't init if it's already been done.
        try:
            self.cursor
            return
        except:
            pass

        st = 1237680000
        et = st + 24 * 3600
        self.dbconn = db.connect()
        self.cursor = self.dbconn.cursor()
        self.sites = dataset.read_sites(self.cursor)
        self.stations, self.name_to_siteid_minus1, self.siteid_minus1_to_name = dataset.read_sites_by_name(self.cursor)
        self.site_up = dataset.read_uptime(self.cursor, st, et)
        self.phasenames, self.phasetimedef = dataset.read_phases(self.cursor)
        self.phaseids = dict(zip(self.phasenames, range(1, len(self.phasenames)+1)))
        self.earthmodel = learn.do_training.load_earth("parameters", self.sites, self.phasenames, self.phasetimedef)
        self.sigmodel = learn.do_training.load_sigvisa("parameters", st, et, "spectral_envelope", self.site_up, self.sites, self.phasenames, self.phasetimedef, load_signal_params = False)

        self.bands = ("freq_2.0_3.0",'freq_0.5_0.7')
        self.chans = ('BHZ', 'BHN', 'BHE')
        self.phases = ('P', 'Pn', 'Pg', 'S', 'Sn', 'Lg')

        self.P_phases = ('P', 'Pn')
        self.S_phases = ('S', 'Sn', 'Lg')

        self.events = dict()

        self.set_dummy_wiggles()

    def __del__():
        self.dbconn.close()

    def phasenames(self, phase_id_minus1_list):
        return [self.phasenames[id] for id in phase_id_minus1_list]

    def arriving_phases(self, event, sta):
        siteid = self.name_to_siteid_minus1[sta] + 1
        phases = [p for p in self.phases if self.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid-1, self.phaseids[p]-1) > 0 ]
        return phases

    def equivalent_channels(self, chan):
        canonical = self.canonical_channel_name[chan]
        equiv = self.__equivalent_channels[canonical]
        return equiv



    def set_dummy_wiggles(self):
        s = self
        for siteid in range(1, len(self.stations)):
            for chan in s.chans:
                c = sigvisa_c.canonical_channel_num(chan)
                for band in s.bands:
                    b = sigvisa_c.canonical_band_num(band)
                    for p in s.phases:
                        pid = s.phaseids[p]
                        s.sigmodel.set_wiggle_process(siteid, b, c, pid, 1, 0.05, np.array([.8,-.2]))


def set_noise_processes(sigmodel, seg):
    for chan in seg.keys():
        c = sigvisa_c.canonical_channel_num(chan)
        for band in seg[chan].keys():
            b = sigvisa_c.canonical_band_num(band)
            siteid = seg[chan][band].stats.siteid
            try:
                arm = seg[chan][band].stats.noise_model
            except KeyError:
#                print "no noise model found for chan %s band %s, not setting.." % (chan, band)
                continue
            sigmodel.set_noise_process(siteid, b, c, arm.c, arm.em.std**2, np.array(arm.params))






