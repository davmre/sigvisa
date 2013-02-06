import numpy as np
import numpy.ma as ma
import scipy.signal
import time, copy, cPickle
from sigvisa import Sigvisa

import obspy.signal.filter
from obspy.core.trace import Trace

from signals.mask_util import *

from database.dataset import *

def load_waveform_from_file(fname):
    wave = cPickle.load(open(fname, 'rb'))
    wave.filtered = dict()
    return wave

def filter_str_extract_band(filter_str):
    band = "broadband"
    for f in filter_str.split(';'):
        if f == "env":
            band = "broadband_envelope"
        if f.startswith('freq'):
            band = f
            break
    return band


class Waveform(object):
    """
    A single waveform trace. Contains methods for generating filtered versions of itself.
    """

    def __init__(self, data=[], srate = 40, stime=0, sta = "AAK", evid=None, segment_stats = None, my_stats = None, **my_stats_entries):
        if isinstance(data, ma.MaskedArray):
            self.data = data
        else:
            a = np.asarray(data)
            m = ma.masked_array(a)
            self.data = m

        # stats that can be shared with other waveforms in a segment
        if segment_stats is not None:
            self.segment_stats = segment_stats
        else:
            npts = len(data)
            etime = stime + npts/float(srate)
            self.segment_stats = {"stime" : stime, "sta": sta, "etime": etime, "len": npts/float(srate), "siteid": Sigvisa().name_to_siteid_minus1[sta]+1}

        # attributes specific to this waveform, e.g. channel or freq band
        if my_stats is not None:
            self.my_stats = my_stats
        else:
            self.my_stats = {"srate" : float(srate), "npts": npts}
            self.my_stats.update(my_stats_entries)

            try:
                fraction_valid = 1 - np.sum([int(v) for v in self.data.mask])/float(len(self.data))
            except TypeError:
                fraction_valid = 1

            self.my_stats.update({"filter_str" : "",
                                     "freq_low": 0.0,
                                     "freq_high": self.my_stats["srate"]/2.0,
                                     "fraction_valid": fraction_valid })

        self.filtered = dict()


    # cost functions for comparing to other waves
    def l1_cost(self, other_wave):
        if self['filter_str'] != other_wave['filter_str'] or self['stime'] != other_wave['stime'] or self['srate'] != other_wave['srate'] or self['npts'] != other_wave['npts']:
            print "error computing distance between:"
            print self
            print other_wave
            raise Exception("waveforms do not align, can't compute L1 distance!")

        return np.sum(np.abs(self.data - other_wave.data))

    def __get_band(self):
        return filter_str_extract_band(self.my_stats['filter_str'])

    def as_obspy_trace(self):
        allstats = dict(self.segment_stats.items() + self.my_stats.items())
        allstats['starttime_unix'] = allstats['stime']
        allstats['sampling_rate'] = allstats['srate']
        allstats['channel'] = allstats['chan']
        allstats['band'] = self['band']


        return Trace(header=allstats, data=self.data)

    def filter(self, filter_str, preserve_intermediate=False):
        """
        Recursive method to generate a filtered waveform.

        Accepts filter strings of the form "freq_2.0_3.0;env;smooth"
        with filters separated by semicolons. The first filter in the
        chain is applied to generate a new waveform ("first_filtered")
        which is then called recursively to apply the other filters.

        Filtered waveforms are cached in the self.filtered dictionary,
        indexed by the first term in their filter string (thus the
        caching works through the recursive calls for more complex
        filters). Thus only the first call to this method with a given
        filter string actually requires any computation.

        If preserve_intermediate is False, the cache keys are the
        entire filter strings rather than just the first terms.
        """

        if filter_str == "":
            return self

        # return result from cache if we have it
        try:
            if filter_str in self.filtered:
                return self.filtered[filter_str]
        except:
            import pdb
            pdb.set_trace()

        # pick out the first filter to apply
        first_filter = ""
        other_filters = filter_str
        while first_filter == "":
            filters = other_filters.split(';')
            first_filter = filters[0]
            other_filters = ';'.join(filters[1:])

        # apply the first filter
        if first_filter in self.filtered:
            first_filtered = self.filtered[first_filter]
        else:
            f, fstats = self.__filter_by_desc(first_filter)
            filtered_data = f(self.data)
            fstats['npts'] = len(filtered_data)
            first_filtered = Waveform(filtered_data, segment_stats=self.segment_stats, my_stats=fstats)

        # then apply the others recursively
        final_filtered = first_filtered.filter(other_filters, preserve_intermediate=preserve_intermediate)

        if preserve_intermediate:
            self.filtered[first_filter] = first_filtered
        self.filtered[filter_str] = final_filtered

        return final_filtered

    def __getitem__(self, key):

        # numeric indices return raw waveform data
        if isinstance(key, slice):
            srate = self['srate']
            stime = self['stime']
            newslice = slice(int((key.start - stime)*srate), int((key.stop - stime)*srate), 1)
            return self.data[newslice]
        # otherwise, we pull from the stats
        elif key in self.segment_stats:
            return self.segment_stats[key]
        elif key in self.my_stats:
            return self.my_stats[key]

        # if we don't have arrivals for this waveform, look them up and cache them
        elif key == "event_arrivals":
            event_arrivals = read_event_detections(cursor=Sigvisa().dbconn.cursor(), evid=self['evid'], stations = [self['sta'],], evtype="leb")
            self.segment_stats['event_arrivals'] = event_arrivals
            return event_arrivals
        elif key == "arrivals": # default to LEB arrivals
            arrivals = read_station_detections(cursor=Sigvisa().dbconn.cursor(), sta=self['sta'], start_time=self['stime'], end_time=self['etime'], arrival_table="leb_arrival")
            self.segment_stats['arrivals'] = arrivals
            return arrivals
        elif key == "arrivals_idcx":
            arrivals = read_station_detections(cursor=Sigvisa().dbconn.cursor(), sta=self['sta'], start_time=self['stime'], end_time=self['etime'], arrival_table="idcx_arrival")
            self.segment_stats['arrivals_idcx'] = arrivals
            return arrivals
        elif key == "band":
            return self.__get_band()
        else:
            raise KeyError("waveform didn't recognized key %s" % key)

    def __filter_by_desc(self, desc):
        """
        Given a string describing a single filter, return the function
        implementing that filter, along with a new copy of my_stats
        reflecting the filter's application.
        """

        pieces = desc.split("_")
        name = pieces[0]

        fstats = self.my_stats.copy()
        fstats["filter_str"] += ";" + desc

        f = None
        if name == "center":
            f = lambda x : ma.masked_array(data = x.data - np.mean(x), mask = x.mask)
        elif name == "log":
            f = lambda x : np.log(x)
        elif name == "hz":
            new_srate = float(pieces[1])
            ratio = self['srate']/new_srate
            rounded_ratio = int(np.round(ratio))
            if np.abs(ratio - rounded_ratio) > 0.00000001:
                raise Exception("new sampling rate %.3f does not evenly divide old rate %.3f" % (new_srate, self['srate']))
            f= lambda x : ma.masked_array(data=scipy.signal.decimate(x, rounded_ratio), mask = x.mask[::rounded_ratio] if isinstance(x.mask, np.ndarray) else False)
            fstats['srate'] = new_srate
        elif name == "env":
            f = lambda x: ma.masked_array(data=obspy.signal.filter.envelope(x.data), mask=x.mask)
        elif name == "smooth":
            if len(pieces) > 1:
                window_len = int(float(pieces[1]) * self['srate'])
            else:
                window_len = 401

            f = lambda x: smooth(x, window_len)
        elif name == "freq":
            low = float(pieces[1])
            high = float(pieces[2])

            f = lambda x : bandpass_missing(x, low, high, self.my_stats['srate'])

            fstats["freq_low"] = low
            fstats["freq_high"] = high
        else:
            raise Exception("unrecognized filter description %s" % desc)
        return f, fstats

    def dump_to_file(self, fname):
        filtered = self.filtered
        self.filtered=None
        cPickle.dump(self, open(fname, 'wb'), protocol=2)
        self.filtered = filtered

    def __str__(self):
        s = "wave pts %d @ %d Hz. " % (self['npts'], self['srate'])

        timestr = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(self['stime']))
        s += "time: %s (%.1f). \nsta: %s, chan %s. " % (timestr, self['stime'], self['sta'], self['chan'])
        s += ', '.join(['%s: %s' % (k, self.my_stats[k]) for k in sorted(self.my_stats.keys()) if k != 'chan'])
        return s

class Segment(object):

    STANDARD_STATS = ["sta", "stime", "etime"]

    filter_order = ['center', 'freq', 'env', 'log', 'smooth', 'hz']

    def __init__(self, waveforms = []):
        self.__chans = dict()
        self.stats = None
        self.filter_str = ""

        self.filtered = dict()

        for wf in waveforms:
            self.addWaveform(wf)


    def addWaveform(self, wf):
        if wf["chan"] in self.__chans:
            raise Exception("this segment already has a waveform for channel %s! (%s)" % (chan))

        if self.stats is None:
            # if this is the first waveform in the signal, check to make sure it has the necessary info
            for stat in Segment.STANDARD_STATS:
                if stat not in wf.segment_stats:
                    raise Exception("waveform is missing metadata %s, cannot add to segment" % (stat))
            self.stats = wf.segment_stats

        else:
            # check to make sure this waveform is compatible with the
            # other waveforms in this segment.
            for stat in Segment.STANDARD_STATS:
                if wf[stat] != self[stat]:
                    raise Exception("waveform conflicts with segment stat %s (%s vs %s)" % (stat, wf[stat], self[stat]))

            # maintain invariant: all waveforms in a segment share the same segment stats object
            wf.segment_stats = self.stats

        # add the waveform to the segment!
        self.__chans[wf["chan"]] = wf

    def with_filter(self, filter_str, force_duplicate=False):
        new_filters = filter_str.split(';')

        # sanity check to avoid creating weird filter strings
        if not force_duplicate:
            s1 = set(new_filters)
            s2 = set(self.filter_str.split(';'))
            duplicates = s1.intersection(s2)
            if len(duplicates) > 0:
                raise Exception("filter str '%s' duplicates a filter already in use (currently '%s')! Use force_duplicate to override this error." % (filter_str, self.filter_str))

        filters = [x for x in (self.filter_str.split(';') + new_filters) if len(x) > 0]
        sorted_filters = []
        for filter_pattern in self.filter_order:
            for filter_name in filters:
                if filter_name.startswith(filter_pattern):
                    sorted_filters.append(filter_name)

        new_filter_str = ';'.join(sorted_filters)
        if new_filter_str in self.filtered:
            new_self = self.filtered[new_filter_str]
        else:
            new_self = copy.copy(self)
            new_self.filter_str = new_filter_str
            self.filtered[new_filter_str] = new_self

        return new_self

    def as_old_style_segment(self, bands=None):
        # return a dictionary mapping channels to obspy Trace objects with attributes squashed appropriately.
        # this copies dictionaries but hopefully not data.

        if bands is None:
            bands = Sigvisa().bands

        old_segment = dict()
        for chan in self.__chans.keys():
            old_segment[chan] = dict()
            for band in bands:
                filtered = self.with_filter(band)
                old_segment[chan][band] = filtered[chan].as_obspy_trace()
        return old_segment

    # return a waveform with stats corresponding to the given channel,
    # but with no data. this is useful when computing the filtered
    # data would take a long time.
    def dummy_chan(self, chan_str):
        chan = self.__chans[chan_str]
        dummy_chan = Waveform(data = np.random.randn(100), segment_stats = self.stats, my_stats=copy.copy(chan.my_stats))
        dummy_chan = dummy_chan.filter(self.filter_str)
        dummy_chan.data = None
        dummy_chan.my_stats['npts'] = chan['npts']
        return dummy_chan

    def get_chans(self):
        return self.__chans.keys()

    def __getitem__(self, key):
        if key in self.__chans:
            return self.__chans[key].filter(self.filter_str)
        elif key == "filter_str":
            return self.filter_str
        elif key in self.stats:
            return self.stats[key]
        elif key in ('arrivals', 'arrivals_idcx', 'event_arrivals'):
            return self.__chans.values()[0][key]
        else:
            raise KeyError("segment didn't recognized key %s" % key)

    def __str__(self):
        timestr = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(self['stime']))
        s = "start: %s (%.1f). sta: %s, chans: %s. " % (timestr, self['stime'], self['sta'], ','.join(sorted(self.__chans.keys())))
        s += "filter: '%s'" % self.filter_str
        return s


def smooth(x,window_len=121,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


#    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),x.data,mode='same')
    return ma.masked_array(y, x.mask)


def bandpass_missing(masked_array, low, high, srate):
    mask = masked_array.mask

    m = obspy.signal.filter.bandpass(masked_array.data, low, high, srate, corners = 4, zerophase=True)

    return ma.masked_array(data=m, mask=mask)

def main():
    data1 = np.random.randn(1000)
    data2 = np.random.randn(1000)
    data3 = data1 + data2
    print s1

if __name__ == "__main__":
    main()
