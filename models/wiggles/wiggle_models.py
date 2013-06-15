import os
import numpy as np

import scipy.io
import cStringIO


from sigvisa import Sigvisa
from sigvisa.learn.train_param_common import load_model
from sigvisa.models import DummyModel
from sigvisa.graph.nodes import Node, ClusterNode
from sigvisa.models.noise.noise_util import get_noise_model


def get_wiggle_param_model_ids(runid, sta, chan, band, phase, model_type, basisid):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sql_query = "select modelid from sigvisa_param_model where model_type = '%s' and site='%s' and chan='%s' and band='%s' and phase='%s' and wiggle_basisid=%d and fitting_runid=%d" % (model_type, sta, chan, band, phase, basisid, runid)
    cursor.execute(sql_query)
    modelids = [m[0] for m in cursor.fetchall()]
    cursor.close()
    return modelids

class WiggleGenerator(object):

    def __init__(self, **kwargs):
        # child classes should set these before calling super()
        assert(self.srate is not None and self.npts is not None and self.logscale is not None and self.basisid is not None)

    def signal_from_features(self, features):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

    def features_from_signal(self, signal):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

    def basis_type(self):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

    def dimension(self):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

    def keys(self):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

    def param_dict_to_array(self, d):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

    def array_to_param_dict(self, a):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

    def save_to_db(self, dbconn):
        raise NotImplementedError("called unimplemented function on WiggleGenerator")

"""
class WiggleModelNode(ClusterNode):

    def __init__(self, wiggle_model_type="dummy", runid=None, phase=None, logscale =False, model_waveform=None, sta=None, chan=None, band=None, label="", parents={}, children=[]):

        # child classes should set these before calling super()
        assert(self.srate is not None and self.npts is not None and self.logscale is not None and self.basisid is not None)

        if model_waveform is not None:
            sta = model_waveform['sta']
            chan = model_waveform['chan']
            band = model_waveform['band']

        nodes = {}
        if wiggle_model_type=="dummy":
            for param in self.keys():
                mNode = Node(model=DummyModel(), parents=parents, children=children, label=param)
                nodes[param] = mNode
        else:
            modelids = get_wiggle_param_model_ids(runid = runid, sta=sta, band=band, chan=chan, phase=phase, model_type = wiggle_model_type, basisid = self.basisid)

            s = Sigvisa()
            cursor = s.dbconn.cursor()
            models = []
            for modelid in modelids:
                sql_query = "select param, model_type, model_fname, modelid from sigvisa_param_model where modelid=%d" % (modelid)
                cursor.execute(sql_query)
                models.append(cursor.fetchone())
            cursor.close()

            # load all relevant models as new graph nodes
            basedir = os.getenv("SIGVISA_HOME")
            for (param, db_model_type, fname, modelid) in sorted(models):
                model = load_model(os.path.join(basedir, fname), db_model_type)
                mNode = Node(model=model, parents=parents, children=children, label='%s' % (param))
                nodes[param] = mNode

        assert(len(nodes) == self.dimension())
        super(WiggleModelNode, self).__init__(label=label, nodes=nodes, parents=parents, children=children)


    from functools32 import lru_cache
    @lru_cache(maxsize=1024)
    def _wiggle_cache(self, feature_tuple):
        return self.signal_from_features(features = feature_tuple)

    def get_wiggle(self, value=None):
        if not value:
            value = self.get_value()
        wiggle = self._wiggle_cache(feature_tuple = tuple(self.param_dict_to_array(value)))
        return wiggle

    def set_params_from_wiggle(self, wiggle):
        params = self.basis_decomposition(wiggle)
        self.set_value(params)

    def get_encoded_params(self):
        f_string = cStringIO.StringIO()
        scipy.io.savemat(f_string, {"params": self.param_dict_to_array(self.get_value())}, oned_as='row')
        ostr = f_string.getvalue()
        f_string.close()
        return ostr

    def set_encoded_params(self, encoded):
        params = self.decode_params(encoded=encoded)
        self.set_value(self.array_to_param_dict(params))

    @staticmethod
    def decode_params(encoded):
        f_string = cStringIO.StringIO(encoded)
        d = scipy.io.loadmat(f_string)
        params = d['params'].flatten()
        f_string.close()
        return params
"""
