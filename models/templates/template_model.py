import numpy as np
import sys
import os
from sigvisa import Sigvisa, NestedDict


from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.learn.train_coda_models import load_model
from sigvisa.signals.common import *

from sigvisa.models.graph import Node, ClusterNode
from sigvisa.models.ttime import TravelTimeModel
from sigvisa.models import DummyModel



class TemplateModelNode(ClusterNode):

    def __init__(self, label="", parents = None, children=None, runid=None, model_type=None, sta=None, chan=None, band=None, phase=None, modelids=None):

        super(TemplateModelNode, self).__init__(label=label, nodes=[], parents=parents, children=children)

        self.phase = phase

        s = Sigvisa()
        cursor = s.dbconn.cursor()

        # get information about the param models to be loaded
        if model_type == "dummy":
            pass
        elif modelids is not None:
            models = []
            for modelid in modelids:
                sql_query = "select param, model_type, model_fname, modelid from sigvisa_template_param_model where modelid=%d" % (modelid)
                cursor.execute(sql_query)
                models.append(cursor.fetchone())

        elif runid is not None and model_type is not None:
            if isinstance(model_type, str):
                model_type_cond = "model_type = '%s'" % model_type
            elif isinstance(model_type, dict):
                model_type_cond = "(" + " or ".join(
                    ["(model_type = '%s' and param = '%s')" % (v, k) for (k, v) in model_type.items()]) + ")"
            else:
                raise Exception("model_type must be either a string, or a dict of param->model_type mappings")

            sql_query = "select param, model_type, model_fname, modelid from sigvisa_template_param_model where %s and site='%s' and chan='%s' and band='%s' and phase='%s' and fitting_runid=%d" % (
                model_type_cond, sta, chan, band, phase, runid)
            cursor.execute(sql_query)
            models = cursor.fetchall()
        else:
            raise Exception("you must specify either a fitting run or a list of template model ids!")

        # load all relevant models as new graph nodes
        self.nodes = []
        self.nodeDict = dict()

        if model_type == "dummy":
            defaults = self.default_param_vals()
            for (i, param) in enumerate(self.params()):
                mNode = Node(model=DummyModel(default_value = defaults[param]), parents=self.parents, children=self.children, label=param)
                self.nodes.append(mNode)
                self.nodeDict[param] = mNode
        else:
            basedir = os.getenv("SIGVISA_HOME")
            for (param, db_model_type, fname, modelid) in models:
                if param == "amp_transfer":
                    param = "coda_height"

                model = load_model(os.path.join(basedir, fname), db_model_type)
                mNode = Node(model=model, parents=self.parents, children=self.children, label=param)

                self.nodes.append(mNode)
                self.nodeDict[param] = mNode

        # also load arrival time models for each phase
        atimeNode = Node(model=TravelTimeModel(sta=sta, phase=phase, arrival_time=True), parents=self.parents, children=self.children, label = 'arrival_time')
        self.nodes.append(atimeNode)
        self.nodeDict['arrival_time'] = atimeNode


    def dimension(self):
        return len(self.params())+1


    def get_mutable_values(self):
        values = []
        for (i, param) in enumerate(('arrival_time',) + self.params()):
            node = self.nodeDict[param]
            if not node.fixed_value:
                values.append(node.get_value())
        return np.array(values)

    def set_mutable_values(self, value):
        i = 0
        for param in ('arrival_time',) + self.params():
            node = self.nodeDict[param]
            if not node.fixed_value:
                node.set_value(value = value[i])
                i += 1

    def get_value(self):
        values = np.zeros((self.dimension(), ))

        for (i, param) in enumerate(('arrival_time',) + self.params()):
            node = self.nodeDict[param]
            values[i] = node.get_value()

        return values

    def set_value(self, value):
        for (i, param) in enumerate(('arrival_time',) + self.params()):
            node = self.nodeDict[param]
            node.set_value(value = value[i])

    def fix_arrival_time(self, fixed=True):
        self.nodeDict['arrival_time'].fixed_value = True

    def log_p(self, value = None):
        lp = 0

        for (i, param) in enumerate(('arrival_time',) + self.params()):
            node = self.nodeDict[param]
            nlp = node.log_p(value = value[i] if value is not None else None)
            # print "logp %f from node %s" % (nlp, param)
            lp += nlp

        return lp




    # return the name of the template model as a string
    @staticmethod
    def model_name():
        raise Exception("abstract class: method not implemented")

    # return a tuple of strings representing parameter names
    @staticmethod
    def params():
        raise Exception("abstract class: method not implemented")

    @staticmethod
    def low_bounds():
        raise Exception("abstract class: method not implemented")

    @staticmethod
    def high_bounds():
        raise Exception("abstract class: method not implemented")


