import numpy as np
import os
import time
from sigvisa.database import db
import sigvisa.database.sites as db_sites
from sigvisa.load_c_components import load_sigvisa, load_earth
import sigvisa.sigvisa_c

from collections import defaultdict
import threading
#from sigvisa.lockfile_pool import get_lock_from_pool

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
    # canonical name for each channel.
    __equivalent_channels = {"BHZ": ["BHZ", "bz"], "BHE": ["BHE", "BH1", "be"],
                             "BHN": ["BHN", "BH2", "bn"],
                             "SHZ": ["SHZ", "sz", "szl"],
                             "SHE": ["SHE", "se"],
                             "SHN": ["SHN", "sn"],
                             "MHZ": ["MHZ", 'mz'], "MHE": ["MHE", "me"], "MHN": ["MHN", "mn"],
                             "EHZ": ["EHZ", 'ez'], "EHE": ["EHE", "ee"], "EHN": ["EHN", "en"],
                             "LHZ": ["LHZ", 'lz'], "LHE": ["LHE", "le"], "LHN": ["LHN", "ln"],
}

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

        self.homedir = os.getenv("SIGVISA_HOME")

        st = 1237680000
        et = st + 24 * 3600

        #self.lock = get_lock_from_pool(os.getenv("SIGVISA_HOME") + "/.db_lock", pool_size=3)
        try:
            self.dbconn = db.connect()
            cursor = self.dbconn.cursor()
        except Exception as e:
            #self.lock.release()
            if "SESSIONS_PER_USER" in str(e):
                raise
                print "too many DB sessions, trying again in 1s..."
                time.sleep(1)
            else:
                raise

        self.canonical_channel_name = dict()
        for chan in self.__equivalent_channels.keys():
            for equiv in self.__equivalent_channels[chan]:
                self.canonical_channel_name[equiv] = chan

        vertical_canonical_channels = ('BHZ', 'SHZ', 'MHZ', 'EHZ', 'LHZ')

        sites = db_sites.read_sites(cursor)
        sitenames, allsites = db_sites.read_all_sites(cursor)
        self.sitenames = sitenames
        self.ref_siteid = dict(zip(sitenames, [int(rsi) for rsi in allsites[:,6]]))
        #self.sitedata = dict(zip(sitenames, allsites))
        #self.stations, self.name_to_siteid_minus1, self.siteid_minus1_to_name = sites.read_sites_by_name(cursor)
        _, _, self.siteid_minus1_to_name = db_sites.read_sites_by_name(cursor)

        # load or create the list of default vertical channels at each station
        dfc_cache_file = os.path.join(self.homedir, "db_cache", "vertical_channels")
        try:
            with open(dfc_cache_file, 'r') as f:
                self.default_vertical_channel = eval(f.read())
        except IOError:
            default_vertical_channel = [(sta, get_sta_default_channel(cursor, sta, vertical_canonical_channels, self.__equivalent_channels)) for sta in sitenames]
            self.default_vertical_channel = dict([(sta, chan) for (sta,chan) in default_vertical_channel if chan is not None])
            with open(dfc_cache_file, 'w') as f:
                f.write(repr(self.default_vertical_channel))



        site_up = db_sites.read_uptime(cursor, st, et)
        self.phasenames, self.phasetimedef = db_sites.read_phases(cursor)
        self.phaseids = dict(
            zip(self.phasenames, range(1, len(self.phasenames) + 1)))
        self.earthmodel = load_earth(os.path.join(self.homedir, "parameters"), sitenames, allsites, self.phasenames, self.phasetimedef)
        self.sigmodel = load_sigvisa(self.earthmodel, os.path.join(self.homedir, "parameters"), site_up, sites, self.phasenames, self.phasetimedef )


#        self.bands = ("freq_2.0_3.0",'freq_0.5_0.7', 'freq_6.0_8.0')
        self.bands = ("freq_2.0_3.0",)
#        self.chans = ('BHZ', 'BHN', 'BHE')
        self.chans = ('BHZ', 'BHN', 'BHE')
        self.phases = ('P', 'Pn', 'Pg', 'PcP', 'pP', 'PKKPbc', 'S', 'Sn', 'ScP', 'Lg')

        self.P_phases = ('P', 'Pn', 'PcP', 'pP', 'PKKPbc')
        self.S_phases = ('S', 'Sn', 'Lg', 'ScP')

        self.events = dict()

        self.global_dict_cache = dict() # cache a mapping of event dictionaries to arrays, used by ParamModels

    def __del__(self):
        self.dbconn.close()
        #self.lock.release()

    def phasenames(self, phase_id_minus1_list):
        return [self.phasenames[pid] for pid in phase_id_minus1_list]

    def arriving_phases(self, event, sta):
        phases = []
        for p in self.phases:
            try:
                tt = self.sigmodel.mean_travel_time(
                         event.lon, event.lat, event.depth, event.time, sta, self.phaseids[p] - 1)
                if tt > 0:
                    phases.append(p)
            except ValueError:
                continue
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

    def get_array_site(self, sta):
        _, _, _, isarr, _, _, ref_site_id = self.earthmodel.site_info(sta, 0)
        ref_site_name = self.siteid_minus1_to_name[ref_site_id-1]
        return ref_site_name

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

    def get_default_sta(self, site):
        if self.earthmodel.site_info(site, 0.0)[3] == 1:
            cursor = self.dbconn.cursor()
            cursor.execute("select refsta from static_site where sta='%s'" % site)
            sta = cursor.fetchone()[0]
            cursor.close()
        else:
            sta = site
        return sta

    def array_default_channel(self, site):
        elems = self.get_array_elements(site)
        chans = [self.default_vertical_channel[elem] for elem in elems]
        return max(set(chans), key=chans.count)

    def sites_to_stas(self, sites, refsta_only=False):
        stas = []
        if refsta_only:
           stas = sites
        else:
            stas = []
            for site in sites:
                if self.is_array_station(site):
                    stas += self.get_array_elements(site)
                else:
                    stas.append(site)
        return stas

def get_sta_default_channel(cursor, sta, canonical_choices, equivalent_channels):

    sql_query = "select chan from static_sitechan where sta='%s'" % sta
    cursor.execute(sql_query)
    chans = [c[0] for c in cursor.fetchall()]
    for canonical in canonical_choices:
        for c in equivalent_channels[canonical]:
            if c in chans:
                return c
    return None
