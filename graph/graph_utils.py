import re
from sigvisa import Sigvisa

from sigvisa.database.dataset import read_event_detections, DET_PHASE_COL

import functools32

def extract_sta_node(node_or_dict, sta):
    try:
        return node_or_dict[sta]
    except AttributeError:
        return node_or_dict


def predict_phases(ev, sta, phases):
    s = Sigvisa()
    if phases == "leb":
        cursor = s.dbconn.cursor()
        predicted_phases = [s.phasenames[id_minus1] for id_minus1 in read_event_detections(cursor=cursor, evid=ev.evid, stations=[sta, ], evtype="leb")[:,DET_PHASE_COL]]
        cursor.close()
    elif phases == "auto":
        predicted_phases = s.arriving_phases(event=ev, sta=sta)
    else:
        predicted_phases = phases
    return predicted_phases

def create_key(param, eid=None, sta=None, phase=None, chan=None, band=None):
    eid = str(int(eid)) if eid is not None else ':'
    sta = sta if sta else ':'
    phase = phase if phase else ':'
    chan = chan if chan else ':'
    band = band if band else ':'
    return "%s;%s;%s;%s;%s;%s" % (eid, phase, sta, chan, band, param)

"""
@functools32.lru_cache(2048)
def get_re(eid, phase, sta, chan, band, param_name):
    return re.compile("(%d|:);(%s|:);(%s|:);(%s|:);(%s|:);%s" % (eid, phase, sta, chan, band, param_name))

def get_parent_value(eid, phase, sta, param_name, parent_values, chan=None, band=None, return_key=False):
    r = get_re(eid, phase, sta, chan, band, param_name)
    for k in parent_values.keys():
        if r.match(k):
            if return_key:
                return (k, parent_values[k])
            else:
                return parent_values[k]
    raise KeyError("could not find parent providing %d;%s;%s;%s;%s;%s" % (eid, phase, sta, chan, band, param_name))
"""
# this is a HACK incorrect version, much faster than the real one though
def get_parent_value(eid, phase, sta, param_name, parent_values, chan=None, band=None, return_key=False):
    k = "%d;%s;%s;%s;%s;%s" % (eid, phase, sta, chan if chan else ":", band if band else ":", param_name)
    try:
        v = parent_values[k]
    except KeyError:
        k = "%d;%s;%s;%s;%s;%s" % (eid, phase, sta, ":", ":", param_name)
        v = parent_values[k]
    if return_key:
        return (k, parent_values[k])
    else:
        return parent_values[k]
