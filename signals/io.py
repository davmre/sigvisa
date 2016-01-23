import time
import sys
import struct
import os
import gzip
import functools32
import numpy as np
import numpy.ma as ma

from optparse import OptionParser
from obspy.core import Trace, Stream, UTCDateTime
import obspy.signal.filter

from sigvisa.database.dataset import *
from sigvisa.signals.common import *
from sigvisa.signals.mask_util import *

from sigvisa import Sigvisa


try:
    from MySQLdb import ProgrammingError
except:
    class ProgrammingError(Exception):
        pass


class MissingWaveform(Exception):
    pass

class EventNotDetected(Exception):
    pass

class ConflictingEvent(Exception):
    pass

def load_event_station_chan(evid, sta, chan, evtype="leb", cursor=None,
                            pre_s = 10, post_s=200, exclude_other_evs=False,
                            phases=None, pad_seconds=20):
    close_cursor = False
    s = Sigvisa()
    if cursor is None:
        cursor = s.dbconn.cursor()
        close_cursor = True

    try:
        arrivals = read_event_detections(cursor, evid, (sta,), evtype=evtype)
        if phases is None:
            if len(arrivals) == 0:
                raise EventNotDetected('no arrivals found for evid %d at station %s' % (evid, sta))
            arrival_times = arrivals[:, DET_TIME_COL]
        else:
            from sigvisa.source.event import get_event
            from sigvisa.models.ttime import tt_predict
            ev = get_event(evid)
            arrphases = s.arriving_phases(ev, sta)
            arrival_times = np.array([ev.time + tt_predict(ev, sta, phase) for phase in phases if phase in arrphases])

        st = np.min(arrival_times) - pre_s
        et = np.max(arrival_times) + post_s

        if exclude_other_evs:
            other_assocs = read_misc_assocs_at_station(cursor, sta, st, et, evid)
            for (other_t, other_evid) in other_assocs:
                if other_t < np.max(arrival_times) + 5.0:
                    raise ConflictingEvent("impossible to fit all phases of event %d at %s, detected at time %.1f-%.1f, due to evid %d arrival at %.1f." % (evid, sta, np.min(arrival_times), np.max(arrival_times), other_evid, other_t))
                else:
                    print "reducing signal window by %.1fs to avoid conflict with %d arrival at %.1f" % (et-other_t+5.0, other_evid, other_t)
                    et = min(et, other_t-5.0)



        print st, et
        wave = fetch_waveform(sta, chan, st, et, pad_seconds=pad_seconds)
        wave.segment_stats['evid'] = evid
        wave.segment_stats['event_arrivals'] = arrivals

    finally:
        if close_cursor:
            cursor.close()

    return wave


def load_event_station(evid, sta, evtype="leb", cursor=None):
    close_cursor = False
    if cursor is None:
        cursor = Sigvisa().dbconn.cursor()
        close_cursor = True

    arrivals = read_event_detections(cursor, evid, (sta,), evtype=evtype)
    arrival_times = arrivals[:, DET_TIME_COL]
    seg = load_segments(cursor, (sta,), np.min(arrival_times) - 10, np.max(arrival_times) + 200)[0]
    seg.stats['evid'] = evid
    seg.stats['event_arrivals'] = arrivals

    if close_cursor:
        cursor.close()

    return seg


def load_segments(cursor, stations, start_time, end_time, chans=None):
    """
    Return a list of waveform segments corresponding to the given channels
    at the given stations over the given time period.
    """

    segments = []

    if chans is None:
        chans = Sigvisa().chans

    # standardize channel names to avoid duplicates
    chans = [Sigvisa().canonical_channel_name[c] if c != 'auto' else 'auto' for c in chans]

    for sta in stations:

        waves = []

        for chan in chans:
            print "loading sta %s chan %s time [%.1f, %.1f]" % (sta, chan, start_time, end_time),
            sys.stdout.flush()
            try:
                wave = fetch_waveform(sta, chan, start_time, end_time, cursor=cursor)
                print " ... successfully loaded."
            except (MissingWaveform, IOError, KeyError) as e:
                print " ... not found, skipping. (%s)" % e
                continue
            waves.append(wave)

        if len(waves) > 0:
            segment = Segment(waves)
            segments.append(segment)

    if len(segments) == 0:
        raise MissingWaveform(
            "couldn't load any waveforms for this segment; check that the data files are in the correct location.")

    return segments


@functools32.lru_cache(maxsize=1024)
def fetch_waveform(station, chan, stime, etime, pad_seconds=20, cursor=None):
    """
    Returns a single Waveform for the given channel at the station in
    the given interval. If there are periods for which data are not
    available, they are marked as missing data.

    Loads data for pad_seconds before and after the true start time, and
    masks the additional data. This is used to absorb filtering
    artifacts.
    """
    s = Sigvisa()

    # scan the waveforms for the given interval
    samprate = None

    stime = stime - pad_seconds
    etime = etime + pad_seconds

    # the global_data array is initialized below once we know the
    # samprate
    global_data = None
    global_stime = stime
    global_etime = etime

    close_cursor = False
    if cursor is None:
        cursor = s.dbconn.cursor()
        close_cursor = True

    if s.earthmodel.site_info(station, stime)[3] == 1:
        cursor.execute("select refsta from static_site where sta='%s'" % station)
        options = [v[0] for v in cursor.fetchall() if v[0] != station]
        selection = options[0]
    else:
        selection = station

    if chan == "auto":
        chan = s.default_vertical_channel[selection]
    chan = s.canonical_channel_name[chan]
    chan_list = s.equivalent_channels(chan)


    # explicitly do BETWEEN queries (with generous bounds) rather than just
    # checking time < etime and endtime > stime, because the latter creates a
    # monstrous slow database join

    MAX_SIGNAL_LEN = 3600 * 8
    sql = "select * from idcx_wfdisc where sta = '%s' and %s and time between %f and %f and endtime between %f and %f" % (
        selection, sql_multi_str("chan", chan_list), stime - MAX_SIGNAL_LEN, etime, stime, etime + MAX_SIGNAL_LEN)
    cursor.execute(sql)
    waveforms = cursor.fetchall()
    if not waveforms:
        raise MissingWaveform("Can't find data for sta %s chan %s time %d"
                              % (station, chan, stime))
    table_description = cursor.description

    if close_cursor:
        cursor.close()


    for waveform_values in waveforms:

        waveform = dict(zip([x[0].lower() for x in table_description], waveform_values))


        assert(samprate is None or samprate == waveform['samprate'])
        if samprate is None:
            samprate = waveform['samprate']

            # initialize the data array full of nans.
            # these will be replace by signals that we load.
            global_data = np.empty((int((global_etime - global_stime) * samprate),))
            global_data.fill(np.nan)

        # at which offset into this waveform should we start collecting samples
        first_offset_time = max(stime - waveform['time'], 0)
        first_offset = int(np.floor(first_offset_time * samprate))
        # how many samples are needed remaining
        load_start_time = waveform['time'] + first_offset_time
        desired_samples = int(np.floor((global_etime - load_start_time) * samprate))
        # how many samples are actually available
        available_samples = waveform['nsamp'] - first_offset
        # grab the available and needed samples
        #try:
        wave = _read_waveform_from_file(waveform, first_offset,
                                        min(desired_samples, available_samples))
        #except IOError:
        #     raise MissingWaveform("Can't find data for sta %s chan %s time %d"
        #                          % (station, chan, stime))

        # copy the data we loaded into the global array
        t_start = max(0, int((waveform['time'] - global_stime) * samprate))
        t_end = t_start + len(wave)
        global_data[t_start:t_end] = wave

#    print "   loaded data from %d to %d (%.1f to %.1f)" % (t_start, t_end, t_start/samprate, t_end/samprate)

        # do we have all the data that we need
        if desired_samples <= available_samples:
            break

        # otherwise move the start time forward for the next file
        stime = waveform['endtime']
        # and adust the end time to ensure that the correct number of samples
        # will be selected in the next file

    masked_data = mirror_missing(ma.masked_invalid(global_data))

    if pad_seconds > 0:
        pad_samples = int(pad_seconds * samprate)
        masked_data[0:pad_samples] = ma.masked
        masked_data[-pad_samples:] = ma.masked



    return Waveform(data=masked_data, sta=selection, stime=global_stime, srate=samprate, chan=chan)

    # return samprate, np.array(data)


def _read_waveform_from_file(waveform, skip_samples, read_samples):
    """
    waveform -- row queried from wfdisc table
    """
    # open the waveform file
    # filename = os.path.join(*(waveform['dir'].split("/")
    #                          + [waveform['dfile']]))
    filename = os.path.join(waveform['dir'], waveform['dfile'])
    try:
        datafile = open(filename, "rb")
    except IOError, e:
        print "cannot open file ", filename,
        # the file could be compressed try .gz extension
        datafile = gzip.open(filename + ".gz")

    assert(waveform['datatype'] in ["s3", "s4"])
    bytes_per_sample = int(waveform['datatype'][-1])

    # seek to the desired offset
    datafile.seek(waveform['foff'] + skip_samples * bytes_per_sample)
    # and read the number of bytes required
    assert(read_samples <= waveform['nsamp'])
    bytes = datafile.read(read_samples * bytes_per_sample)
    datafile.close()

    # now convert the bytes into an array of integers

    data = np.ndarray((read_samples,), int)

    if waveform['datatype'] == "s4":
        data = struct.unpack(">%di" % read_samples, bytes)

    else:
        # s3
        for dest in xrange(read_samples):
            src = dest * 3

            # if the first byte's MSB is set then add an FF to the number
            first = struct.unpack("B", bytes[src])[0]
            if first >= 128:
                data[dest] = struct.unpack(">i", "\xff" + bytes[src:src + 3])[0]
            else:
                data[dest] = struct.unpack(">i", "\x00" + bytes[src:src + 3])[0]

    # convert the raw values into nm (nanometers)
    calib = float(waveform['calib'])
    return [float(x) * calib for x in data]
