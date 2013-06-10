import numpy as np
import os
import time
from sigvisa.database import db
import sigvisa.database.sites as db_sites
from sigvisa.load_c_components import load_sigvisa, load_earth
import sigvisa.sigvisa_c

import threading

class NestedDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return self.setdefault(key, NestedDict())

class BadParamTreeException(Exception):
    pass


class Sigvisa(threading.local):

    # some channels are referred to by multiple names, i.e. the
    # north-south channel can be BHN or BH2.  here we define a
    # canonical name for each channel. the sigvisa.equivalent_channels(chan)
    # method (defined below) returns a list of all channel names
    # equivalent to a particular channel.
    canonical_channel_name = {"BHZ": "BHZ", "BHN": "BHN", "BHE": "BHE",
                              "BH1": "BHE", "BH2": "BHN",
                              "SHZ": "SHZ", "sz": "SHZ",
                              "SHN": "SHN", "sn": "SHN",
                              "SHE": "SHE", "se": "SHE",}

    # defined only for canonical channels
    __equivalent_channels = {"BHZ": ["BHZ"], "BHE": ["BHE", "BH1"],
                             "BHN": ["BHN", "BH2"],
                             "SHZ": ["SHZ", "sz"],
                             "SHE": ["SHE", "se"],
                             "SHN": ["SHN", "sn"],}

    # singleton pattern -- only initialize once
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Sigvisa, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):

        # enforcing the singleton pattern: don't init if it's already been
        # done.
        try:
            self.dbconn
            return
        except:
            pass

        st = 1237680000
        et = st + 24 * 3600

        while True:
            try:
                self.dbconn = db.connect()
                cursor = self.dbconn.cursor()
                break
            except Exception as e:
                if "SESSIONS_PER_USER" in str(e):
                    print "too many DB sessions, trying again in 1s..."
                    time.sleep(1)
                else:
                    raise

        sites = db_sites.read_sites(cursor)
        sitenames, allsites = db_sites.read_all_sites(cursor)
        self.sitenames = sitenames
        self.ref_siteid = dict(zip(sitenames, [int(rsi) for rsi in allsites[:,6]]))
        #self.sitedata = dict(zip(sitenames, allsites))
        #self.stations, self.name_to_siteid_minus1, self.siteid_minus1_to_name = sites.read_sites_by_name(cursor)
        site_up = db_sites.read_uptime(cursor, st, et)
        self.phasenames, self.phasetimedef = db_sites.read_phases(cursor)
        self.phaseids = dict(
            zip(self.phasenames, range(1, len(self.phasenames) + 1)))
        self.earthmodel = load_earth(os.path.join(os.getenv(
            "SIGVISA_HOME"), "parameters"), sitenames, allsites,
            self.phasenames, self.phasetimedef)
        self.sigmodel = load_sigvisa(self.earthmodel, os.path.join(os.getenv(
            "SIGVISA_HOME"), "parameters"),
            site_up, sites,
            self.phasenames, self.phasetimedef )

#        self.bands = ("freq_2.0_3.0",'freq_0.5_0.7', 'freq_6.0_8.0')
        self.bands = ("freq_2.0_3.0",)
#        self.chans = ('BHZ', 'BHN', 'BHE')
        self.chans = ('BHZ', 'BHN', 'BHE')
        self.phases = ('P', 'Pn', 'Pg', 'PcP', 'S', 'Sn', 'Lg')

        self.P_phases = ('P', 'Pn', 'PcP')
        self.S_phases = ('S', 'Sn', 'Lg')

        self.events = dict()

    def __del__(self):
        self.dbconn.close()

    def phasenames(self, phase_id_minus1_list):
        return [self.phasenames[pid] for pid in phase_id_minus1_list]

    def arriving_phases(self, event, sta):
        phases = [p for p in self.phases if self.sigmodel.mean_travel_time(
            event.lon, event.lat, event.depth, event.time, sta, self.phaseids[p] - 1) > 0]
        return phases

    def equivalent_channels(self, chan):
        canonical = self.canonical_channel_name[chan]
        equiv = self.__equivalent_channels[canonical]
        return equiv


    def band_name(self, low_band=None, high_band=None):
        low_match = [band for band in self.bands if np.abs(float(band.split(
            '_')[1]) - low_band) < 0.01] if low_band is not None else self.bands
        high_match = [band for band in low_match if np.abs(
            float(band.split('_')[2]) - high_band) < 0.01] if high_band is not None else low_match
        return high_match[0]

    def is_array_station(self, sta):
        return self.earthmodel.site_info(sta, 0)[3] == 1

    def get_array_elements(self, sta):
        if not self.is_array_station(sta):
            raise Exception("cannot get elements of non-array station %s" % sta)
        else:
            ref_siteid = self.earthmodel.site_info(sta, 0)[6]
            cursor = self.dbconn.cursor()
            sql_query = "select distinct s.refsta from static_site s, static_siteid sid where sid.id=%d and s.sta=sid.sta" % ref_siteid
            cursor.execute(sql_query)
            refsta = cursor.fetchone()[0]
            sql_query = "select distinct s.sta from static_site s where s.refsta='%s' and s.statype='ss'" % refsta
            cursor.execute(sql_query)
            elements = cursor.fetchall()
            return [s[0] for s in elements]
            cursor.close()
