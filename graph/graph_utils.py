import re
from sigvisa import Sigvisa

from sigvisa.database.dataset import read_event_detections, DET_PHASE_COL



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

def get_parent_value(eid, phase, sta, param_name, parent_values, chan=None, band=None, return_key=False):
    r = re.compile("(%d|:);(%s|:);(%s|:);(%s|:);(%s|:);%s" % (eid, phase, sta, chan, band, param_name))
    for k in parent_values.keys():
        if r.match(k):
            if return_key:
                return (k, parent_values[k])
            else:
                return parent_values[k]
    raise KeyError("could not find parent providing %d;%s;%s;%s;%s;%s" % (eid, phase, sta, chan, band, param_name))
