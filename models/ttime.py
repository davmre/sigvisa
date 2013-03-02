import numpy as np

from sigvisa.models import ConditionalDist

from sigvisa import Sigvisa
from sigvisa.source.event import Event


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

    def predict(self, cond):
        if isinstance(cond, Event):
            event = cond
        else:
            assert(len(cond) == 1)
            event = cond.values()[0]

        meantt = self.s.sigmodel.mean_travel_time(event.lon, event.lat, event.depth, self.siteid - 1, self.phaseid - 1)
        if self.atime:
            return meantt + event.time
        else:
            return meantt

    def sample(self, cond):
        if isinstance(cond, Event):
            event = cond
        else:
            assert(len(cond) == 1)
            event = cond.values()[0]

        meantt = self.predict(event)

        # sample from a Laplace distribution:
        U = np.random.random() - .5
        tt = meantt - self.ttscale * np.sign(U) * np.log(1 - 2 * np.abs(U))
        return tt

    def log_p(self, x, cond):
        if isinstance(cond, Event):
            event = cond
        else:
            assert(len(cond) == 1)
            event = cond.values()[0]

        meantt = self.predict(event)
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
