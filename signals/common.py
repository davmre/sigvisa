import numpy as np
import time, copy

class Waveform(object):
    """
    A single waveform trace. Contains methods for generating filtered versions of itself.
    """

    def __init__(self, data, srate = None, stime=None, sta = None, segment_stats = None, my_stats = None, **my_stats_entries):
        self.data = np.asarray(data)

        # stats that can be shared with other waveforms in a segment
        if segment_stats is not None:
            self.segment_stats = segment_stats
        else:
            npts = len(data)
            etime = stime + npts/float(srate)
            self.segment_stats = {"srate" : float(srate), "stime" : stime, "sta": sta, "npts": npts, "etime": etime}

        # attributes specific to this waveform, e.g. channel or freq band
        if my_stats is not None:
            self.my_stats = my_stats
        else:
            self.my_stats = my_stats_entries 
            my_stats_entries.update({"filtering" : "", 
                                     "freq_low": 0.0, 
                                     "freq_high": self.segment_stats["srate"]/2.0})

        self.filtered = dict()

    def as_obspy_trace(self):
        allstats = dict(self.segment_stats.items() + self.my_stats.items())
        return Trace(header=allstats, data=self.data)

    def filter(self, filter_str, preserve_intermediate=True):
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

        filters = filter_str.split(';')
        first_filter = filters[0]
        other_filters = ';'.join(filters[1:])

        f, fstats = self.__filter_by_desc(first_filter)
        if filter_str not in self.filtered:
            filtered_data = f(self.data)
            filtered_wf = Waveform(filtered_data, my_stats=fstats)
            first_filtered = filtered_wf

        if preserve_intermediate:
             self.filtered[first_filter] = first_filtered
        final_filtered = first_filtered.filter(other_filters)
        if not preserve_intermediate:
            self.filtered[filter_str] = final_filtered
        return final_filtered

    def __getitem__(self, key):

        # numeric indices return raw waveform data
        if isinstance(key, slice):
            return self.data[key]

        # otherwise, we pull from the stats
        elif key in self.segment_stats:
            return self.segment_stats[key]
        elif key in self.my_stats:
            return self.my_stats[key]
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
        fstats["filtering"] += ";" + filter_str         

        f = None
        if name == "env":
            f = lambda x: obspy.signal.filter.envelope(x)
        elif name == "smooth":
            window_len = int(pieces[1])
            f = lambda x: smooth(x, window_len)
        elif name == "freqband":
            low = float(pieces[1])
            high = float(pieces[2])
            f = lambda x : obspy.signal.filter.bandpass(x, low, high, self.segment_stats['srate'], corners = 4, zerophase=True)            
            fstats["freq_low"] = low
            fstats["freq_high"] = high
        else:
            raise Exception("unrecognized filter description %s" % desc)
        return f, fstats

    def __str__(self):
        s = "wave pts %d @ %d Hz. " % (self['npts'], self['srate'])

        timestr = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(self['stime']))
        s += "time: %s. sta: %s, chan %s. " % (timestr, self['sta'], self['chan'])
        s += ', '.join(['%s: %s' % (k, self.my_stats[k]) for k in sorted(self.my_stats.keys()) if k != 'chan'])
        return s

class Segment(object):

    STANDARD_STATS = ["srate", "sta", "stime", "etime", "npts"]

    def __init__(self, waveforms = []):
        self.chans = dict()
        self.stats = None
        self.filter_str = ""

        for wf in waveforms:
            self.addWaveform(wf)

    def addWaveform(self, wf):
        if wf["chan"] in self.chans:
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

        # add the waveform to the segment!
        self.chans[wf["chan"]] = wf

    def filter(self, filter_str):
        new_self = copy.copy(self)
        new_self.filter_str = self.filter_str + ';' + filter_str
        return new_self

    def as_old_style_segment(self):
        # return a dictionary mapping channels to obspy Trace objects with attributes squashed appropriately. 
        # this copies dictionaries but hopefully not data.

        old_segment = dict()
        for chan in self.chans.keys():
            old_segment[chan] = self.chans[chan].filter(self.filter_str).as_obspy_trace()
        return old_segment

    def __getitem__(self, key):
        if key in self.chans:
            return self.chans[key].filter(self.filter_str)
        elif key == "filter":
            return self.filter_str
        elif key in self.stats:
            return self.stats[key]
        else:
            raise KeyError("segment didn't recognized key %s" % key)

    def __str__(self):
        s = "wave pts %d @ %d Hz. " % (self['npts'], self['srate'])

        timestr = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(self['stime']))
        s += "time: %s. sta: %s, chans: %s. " % (timestr, self['sta'], ','.join(sorted(self.chans.keys())))
        s += 'filter: %s' % self.filter_str
        return s


def smooth(x,window_len=11,window='hanning'):
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


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def main():
    data1 = np.random.randn(1000)
    data2 = np.random.randn(1000)
    data3 = data1 + data2





    print s1

if __name__ == "__main__":
    main()
