import numpy as np
import sys
import os
from sigvisa import Sigvisa, NestedDict


from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.learn.train_param_common import load_model
from sigvisa.signals.common import *

from sigvisa.graph.nodes import Node, ClusterNode
from sigvisa.models.ttime import TravelTimeModel
from sigvisa.models import DummyModel


def get_template_param_model_ids(runid, sta, chan, band, phase, model_type):
    s = Sigvisa()
    cursor = s.dbconn.cursor()


    if isinstance(model_type, str):
        model_type_cond = "model_type = '%s'" % model_type
    elif isinstance(model_type, dict):
        model_type_cond = "(" + " or ".join(
            ["(model_type = '%s' and param = '%s')" % (v, k) for (k, v) in model_type.items()]) + ")"
    else:
        raise Exception("model_type must be either a string, or a dict of param->model_type mappings")

    sql_query = "select modelid from sigvisa_param_model where %s and site='%s' and chan='%s' and band='%s' and phase='%s' and fitting_runid=%d" % (model_type_cond, sta, chan, band, phase, runid)
    cursor.execute(sql_query)
    modelids = [m[0] for m in cursor.fetchall()]
    cursor.close()

    return modelids



class TemplateModelNode(ClusterNode):

    def __init__(self, label="", parents = None, children=None, runid=None, model_type=None, sta=None, chan=None, band=None, phase=None, modelids=None, dummy_fallback=False):


        s = Sigvisa()
        cursor = s.dbconn.cursor()

        # ensure we have a list of modelids, and know the station/channel/band/phase
        if modelids:
            assert(len(modelids) > 0 and sta is None and chan is None and band is None and phase is None)
            sql_query = "select site, chan, band, phase from sigvisa_param_model where modelid=%d" % (modelids[0],)
            cursor.execute(sql_query)
            sta, chan, band, phase = cursor.fetchone()
        elif model_type != "dummy":
            modelids = get_template_param_model_ids(runid = runid, sta=sta, band=band, chan=chan, phase=phase, model_type = model_type)

        # load all relevant models as new graph nodes
        nodes = dict()

        # construct arrival time model
        atimeNode = Node(model=TravelTimeModel(sta=sta, phase=phase, arrival_time=True),
                         label = 'arrival_time')
        atimeNode.modelid = None
        nodes['arrival_time'] = atimeNode

        defaults = self.default_param_vals()
        if model_type == "dummy":
            for (i, param) in enumerate(self.params()):
                mNode = Node(model=DummyModel(default_value = defaults[param]),
                             label=param)
                mNode.modelid = None
                nodes[param] = mNode
        else:
            for modelid in modelids:
                sql_query = "select param, model_type, model_fname from sigvisa_param_model where modelid=%d" % (modelid)
                cursor.execute(sql_query)
                param, db_model_type, fname = cursor.fetchone()

                basedir = os.getenv("SIGVISA_HOME")
                if param == "amp_transfer":
                    param = "coda_height"
                model = load_model(os.path.join(basedir, fname), db_model_type)
                mNode = Node(model=model, label=param)
                mNode.modelid = modelid

                nodes[param] = mNode

        # ensure that the node list ordering matches the parameter list
        for param in ('arrival_time',) + self.params():
            if not param in nodes:
                if dummy_fallback:
                    mNode = Node(model=DummyModel(default_value = defaults[param]),
                                 label=param)
                    mNode.modelid = None
                    nodes[param] = mNode
                    print "warning: falling back to dummy model for %s, %s, %s phase %s param %s" % (sta, chan, band, phase, param)
                else:
                    raise KeyError('no template model found for %s, %s, %s phase %s param %s' % (sta, chan, band, phase, param))
        self.phase = phase

        super(TemplateModelNode, self).__init__(label=label, nodes=nodes, parents=parents, children=children)

    def get_modelids(self):
        modelids = {k : self._nodes[k].modelid for k in self.keys() if self._nodes[k].modelid is not None}
        return modelids

    # return the name of the template model as a string
    @staticmethod
    def model_name():
        raise Exception("abstract class: method not implemented")

    # return a tuple of strings representing parameter names
    @staticmethod
    def params():
        raise Exception("abstract class: method not implemented")


