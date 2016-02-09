import tarfile
import shutil
import os

import numpy as np

from sigvisa.utils.fileutils import mkdir_p

def load_serialized_from_file(tgzfile):
    tf = tarfile.TarFile.open(tgzfile, mode="r:gz")

    fnames  = tf.getnames()

    uadicts_by_sta = dict()
    for fname in fnames:
        if not fname.startswith("ua"): continue
        sta, chan, band = fname[3:-4].split(";")
        f = tf.extractfile(fname)
        uadicts = [eval(l) for l in f.readlines()]
        uadicts_by_sta[(sta, chan, band)] = uadicts
        f.close()

    ev_f = tf.extractfile("evstate.txt")
    evdicts = [eval(l) for l in ev_f.readlines()]
    ev_f.close()
    tf.close()

    return evdicts, uadicts_by_sta

def save_serialized_to_file(basename, evdicts, uadicts_by_sta, config_str=None):
    mkdir_p(basename)

    tf = tarfile.TarFile.open(basename+".tgz", mode="w:gz")

    fname = "evstate.txt"
    with open(os.path.join(basename, fname), 'w') as f:
        for evdict in evdicts:
            f.write(repr(evdict) + "\n")
    tf.add(os.path.join(basename, fname), arcname=fname)

    for (sta, chan, band), uadicts in uadicts_by_sta.items():
        fname = "ua_%s;%s;%s.txt" % (sta, chan, band)
        with open(os.path.join(basename, fname), 'w') as f:
            for uadict in uadicts:
                f.write(repr(uadict) + "\n")
        tf.add(os.path.join(basename, fname), arcname=fname)

    if config_str is not None:
        fname = "config.txt"
        with open(os.path.join(basename, fname), 'w') as f:
            f.write(config_str)
        tf.add(os.path.join(basename, fname), arcname=fname)

    tf.close()

    shutil.rmtree(basename)

    print "wrote to", basename+".tgz"

def merge_serializations(*serialization_periods):
    """
    Given: list of serialized states from inference periods with defined start and end times. 
    Returns: a single serialized inference state, combining the within-time-bounds components 
             of the state from all periods. 
    """

    evdicts = []
    uadicts_by_sta = defaultdict(list)
    
    for p_evdicts, p_uadicts_by_sta, stime, etime in serialization_periods:
        filtered_evdicts, filtered_uadicts_by_sta = extract_time_period(p_evdicts, p_uadicts_by_sta, stime, etime)
        evdicts += filtered_evdicts
        for (sta, chan, band), uadicts in filtered_uadicts_by_sta.items():
            uadicts_by_sta[(sta, chan, band)] += uadicts

    return evdicts, uadicts_by_sta

def extract_time_period(evdicts, uadicts_by_sta, stime, etime):
    """
    Given: a serialized inference state, and start and end times (unix timestamps)
    Returns: all events and uatemplates that occur during the given time period.
    """

    filtered_evdicts = []
    filtered_uadicts_by_sta = defaultdict(list)

    for (sta, chan, band), uadicts in uadicts_by_sta:
        for d in uadicts:
            t = d["arrival_time"]
            if stime <= t < etime:
                filtered_uadicts_by_sta[(sta, band, chan)].append(d)

    for d in evdicts:
        t = d["time"]
        if stime <= t < etime:
            filtered_evdicts.append(d)
    
    return filtered_evdicts, filtered_uadicts_by_sta
