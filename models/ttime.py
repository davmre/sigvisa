import numpy as np

from sigvisa.models import Distribution

from sigvisa import Sigvisa
from sigvisa.source.event import Event



class TravelTimeModel(Distribution):

    def __init__(self, sta, phase, arrival_time=False):
        s = Sigvisa()
        self.sta = sta
        self.phase = phase

        self.sta = sta
        self.ref_siteid = s.ref_siteid[sta]
        self.phaseid = s.phaseids[phase]

        # peak of a laplace distribution is 1/2b, where b is the
        # scale param, so (HACK ALERT) we can recover b by
        # evaluating the density at the peak
        self.ttscale = 2.0 / np.exp(s.sigmodel.arrtime_logprob(0, 0, 0, self.ref_siteid - 1, self.phaseid - 1))

        self.atime = arrival_time

    @staticmethod
    def _get_lldt(cond):
        if isinstance(cond, dict) and len(cond) == 1:
            cond = cond.values()[0]

        if isinstance(cond, Event):
            lon, lat, depth, t = cond.lon, cond.lat, cond.depth, cond.time
        elif isinstance(cond, dict):
            lon, lat, depth, t = cond['lon'], cond['lat'], cond['depth'], cond['time']
        else:
            raise ValueError("don't know how to extract lon, lat, depth from %s" % cond)
        return lon, lat, depth, t

    def predict(self, cond):
        s = Sigvisa()
        lon, lat, depth, t = self._get_lldt(cond)

        if t < 1:
            import pdb; pdb.set_trace()

        meantt = s.sigmodel.mean_travel_time(lon, lat, depth, t, self.sta, self.phaseid - 1)
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
        s = Sigvisa()
        meantt = self.predict(cond)
        ll = s.sigmodel.arrtime_logprob(x, meantt, 0, self.ref_siteid - 1, self.phaseid - 1)
        return ll


def tt_predict(event, sta, phase):
    s = Sigvisa()
    phaseid = s.phaseids[phase]

    if event.time < 1:
        import pdb; pdb.set_trace()


    meantt = s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, event.time, sta, phaseid - 1)
    return meantt

def tt_log_p(x, event, sta, phase):
    s = Sigvisa()
    phaseid = s.phaseids[phase]
    ref_siteid = s.ref_siteid[sta]

    if event.time < 1:
        import pdb; pdb.set_trace()


    meantt = s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, event.time, sta, phaseid - 1)
    ll = s.sigmodel.arrtime_logprob(x, meantt, 0, ref_siteid-1, phaseid - 1)
    return ll
