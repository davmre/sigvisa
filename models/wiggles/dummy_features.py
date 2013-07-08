import numpy as np
from sigvisa import Sigvisa
from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models.wiggles.wiggle_models import WiggleGenerator
from sigvisa.models import DummyModel
from sigvisa.models.distributions import Uniform, Gaussian

import scipy.weave as weave
from scipy.weave import converters


class DummyFeatureGenerator(WiggleGenerator):

    def __init__(self, npts, srate, envelope, logscale=False, basisid=None, family_name=None, **kwargs):

        self.srate = srate
        self.npts = npts
        assert( self.npts % 2 == 0 )

        self.len_s = float(npts-1) / srate
        self.x = np.linspace(0, self.len_s, self.npts)

        self.logscale = logscale
        print "created with envelope", envelope
        self.envelope = envelope

        self.family_name = family_name
        self.basisid = basisid
        if basisid is None:
            s = Sigvisa()
            self.save_to_db(s.dbconn)

        # load template params and initialize Node stuff
        super(DummyFeatureGenerator, self).__init__(**kwargs)

    def signal_from_features(self, features):
        signal = np.zeros((self.npts,))
        if self.envelope:
            if self.logscale:
                signal = np.exp(signal)
            else:
                signal += 1

        return signal

    def features_from_signal(self, signal):
        return {}

    def basis_type(self):
        return "dummy"

    def dimension(self):
        return 0

    def params(self):
        return ()

    def param_dict_to_array(self, d):
        return ()

    def array_to_param_dict(self, a):
        return {}

    def save_to_db(self, dbconn):
        assert(self.basisid is None)
        sql_query = "insert into sigvisa_wiggle_basis (srate, logscale, family_name, basis_type, npts, dimension, envelope) values (%f, '%s', '%s', '%s', %d, %d, '%s')" % (self.srate, 't' if self.logscale else 'f', self.family_name, self.basis_type(), self.npts, self.dimension(), 't' if self.envelope else 'f')
        self.basisid = execute_and_return_id(dbconn, sql_query, "basisid")
        return self.basisid

    def unassociated_model(self, param):
        raise AttributeError("cannot return unassociated param model because DummyFeatureGenerator has no params!")
        return DummyModel(default_value=0.0)
