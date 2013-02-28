

from sigvisa import Sigvisa

class EventPriorModel(Distribution):

    def log_p(self, x):
        s = Sigvisa()
        loc_lp = s.sigmodel.event_location_prior_logprob(x.lon, x.lat, x.depth)
        mb_lp = s.sigmodel.event_mag_prior_logprob(x.mb)
        source_lp = 0 # eventually I should have a real model for event source type

        return loc_lp + mb_lp + source_lp

