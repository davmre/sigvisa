import os
import numpy as np

from sigvisa import Sigvisa
from sigvisa.models import DummyModel
from sigvisa.models.graph import Node, ClusterNode
from sigvisa.models.wiggles import load_basis
from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.models.wiggles import load_basis_by_family


def get_wiggle_param_model_ids(runid, sta, chan, band, phase, model_type, basisid):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sql_query = "select wpmid from sigvisa_wiggle_param_model where model_type = '%s' and site='%s' and chan='%s' and band='%s' and phase='%s' and basisid=%d and fitting_runid=%d" % (model_type, sta, chan, band, phase, basisid, runid)
    cursor.execute(sql_query)
    wpmids = cursor.fetchall()
    return wpmids

class WiggleModelNode(ClusterNode):

    def __init__(self, basis_family, wiggle_model_type, model_waveform, phase, runid, label="", parents={}, children=[]):

        super(WiggleModelNode, self).__init__(label=label, nodes=[], parents=parents, children=children)

        srate = model_waveform['srate']
        sta = model_waveform['sta']
        chan = model_waveform['chan']
        band = model_waveform['band']

        featurizer, basisid = load_basis_by_family(family_name = basis_family, runid = runid, sta=sta, chan=chan, band=band, phase=phase, srate=srate)
        self.atime_offset_seconds = featurizer.lead_time

        self.nodes = []
        if wiggle_model_type=="dummy":
            for param in range(featurizer.dimension()):
                param_str = "%03d" % param
                mNode = Node(model=DummyModel(), parents=parents, children=children, label=param_str)
                self.nodes.append(mNode)
        else:
            wpmids = get_wiggle_param_model_ids(runid = runid, sta=sta, band=band, chan=chan, phase=phase, model_type = wiggle_model_type, basisid = basisid)

            s = Sigvisa()
            cursor = s.dbconn.cursor()
            models = []
            for wpmid in wpmids:
                sql_query = "select param, model_type, model_fname, wpmid, basisid from sigvisa_wiggle_param_model where wpmid=%d" % (wpmid)
                cursor.execute(sql_query)
                models.append(cursor.fetchone())
            cursor.close()

            # load all relevant models as new graph nodes

            basedir = os.getenv("SIGVISA_HOME")
            for (param, db_model_type, fname, wpmid, wpm_basisid) in sorted(models):
                model = load_model(os.path.join(basedir, fname), db_model_type)
                mNode = Node(model=model, parents=parents, children=children, label='%s' % (param))
                self.nodes.append(mNode)

        assert(len(self.nodes) == featurizer.dimension())
        self.featurizer = featurizer

    def get_wiggle(self, npts):
        wiggle = self.featurizer.signal_from_features(features = self.get_value(), npts=npts)
        return wiggle

