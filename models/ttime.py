import numpy as np

from sigvisa.models import ConditionalDist

from sigvisa import Sigvisa
from sigvisa.source.event import Event
from sigvisa.models.ev_prior import EV_LON, EV_LAT, EV_DEPTH, EV_TIME



class TravelTimeModel(ConditionalDist):

    def __init__(self, sta, phase, arrival_time=False):
        self.s = Sigvisa()
        self.sta = sta
        self.phase = phase

        self.siteid = self.s.name_to_siteid_minus1[sta] + 1
        self.phaseid = self.s.phaseids[phase]

        # peak of a laplace distribution is 1/2b, where b is the
        # scale param, so (HACK ALERT) we can recover b by
        # evaluating the density at the peak
        self.ttscale = 2.0 / np.exp(self.s.sigmodel.arrtime_logprob(0, 0, 0, self.siteid - 1, self.phaseid - 1))

        self.atime = arrival_time

    @staticmethod
    def _get_lldt(cond):
        if isinstance(cond, dict):
            assert( len(cond) == 1 )
            cond = cond.values()[0]

        if isinstance(cond, Event):
            lon, lat, depth, t = cond.lon, cond.lat, cond.depth, cond.time
        elif isinstance(cond, np.ndarray):
            lon, lat, depth, t = cond[EV_LON], cond[EV_LAT], cond[EV_DEPTH], cond[EV_TIME]
        else:
            raise ValueError("don't know how to extract lon, lat, depth from %s" % cond)
        return lon, lat, depth, t

    def predict(self, cond):
        lon, lat, depth, t = self._get_lldt(cond)
        meantt = self.s.sigmodel.mean_travel_time(lon, lat, depth, self.siteid - 1, self.phaseid - 1)
        if self.atime:
            return meantt + t
        else:
            return meantt

    def sample(self, cond):
        meantt = self.predict(cond)

        # sample from a Laplace distribution:
        U = np.random.random() - .5
        tt = meantt - self.ttscale * np.sign(U) * np.log(1 - 2 * np.abs(U))
        return tt

    def log_p(self, x, cond):
        meantt = self.predict(cond)
        ll = self.s.sigmodel.arrtime_logprob(x, meantt, 0, self.siteid - 1, self.phaseid - 1)
        return ll




def tt_predict(event, sta, phase):
    s = Sigvisa()
    siteid = s.name_to_siteid_minus1[sta] + 1
    phaseid = s.phaseids[phase]

    meantt = s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid - 1, phaseid - 1)
    return meantt

def tt_log_p(x, event, sta, phase):
    s = Sigvisa()
    siteid = s.name_to_siteid_minus1[sta] + 1
    phaseid = s.phaseids[phase]

    meantt = s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, siteid - 1, phaseid - 1)
    ll = s.sigmodel.arrtime_logprob(x, meantt, 0, siteid - 1, phaseid - 1)
    return ll
