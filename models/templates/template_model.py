import numpy as np
import sys
import os
from sigvisa import Sigvisa, NestedDict


from sigvisa.models.noise.noise_util import get_noise_model
from sigvisa.learn.train_coda_models import load_model
from sigvisa.signals.common import *

from sigvisa.models.graph import Node, ClusterNode
from sigvisa.models.ttime import TravelTimeModel




def TemplateModelNode(ClusterNode):

    def __init__(self, label="", parents = {}, children=[], runid=None, model_type=None, sta=None, chan=None, band=None, phase=None, modelids=None):

        super(self, ClusterNode).__init__(label=label, nodes=[], parents=parents, children=children)

        s = Sigvisa()
        cursor = s.dbconn.cursor()

        # get information about the param models to be loaded
        if modelids is not None:
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

                sql_query = "select param, model_type, model_fname, modelid from sigvisa_template_param_model where %s and sta='%s' and chan='%s' and band='%s' and phase='%s' and fitting_runid=%d" % (
                model_type_cond, sta, chan, band, phase, runid)
            cursor.execute(sql_query)
            models = cursor.fetchall()
        else:
            raise Exception("you must specify either a fitting run or a list of template model ids!")

        # load all relevant models as new graph nodes
        self.nodes = []
        self.nodeDict = NestedDict()
        basedir = os.getenv("SIGVISA_HOME")
        for (param, db_model_type, fname, modelid) in models:
            if param == "amp_transfer":
                param = "coda_height"

            model = load_model(os.path.join(basedir, fname), db_model_type)
            mNode = Node(model=model, parents=parents, children=children, label=param)

            self.nodes.append(mNode)
            self.nodeDict[param] = mNode

        # also load arrival time models for each phase
        atimeNode = Node(model=TravelTimeModel(sta=sta, phase=phase, arrival_time=True), parents=parents, label = 'arrival_time')
        self.nodeDict['arrival_time'] = atimeNode

    def get_value(self):
        vals = np.zeros((len(self.params())+1, ))

        for (i, param) in enumerate(('arrival_time',) + self.params()):
            node = self.nodeDict[param]
            assert( isinstance(node, graph.Node) )
            vals[i] = node.get_value()

        return vals

    # return the name of the template model as a string
    @staticmethod
    def model_name():
        raise Exception("abstract class: method not implemented")

    # return a tuple of strings representing parameter names
    @staticmethod
    def params():
        raise Exception("abstract class: method not implemented")

    @staticmethod
    def low_bounds(phases):
        raise Exception("abstract class: method not implemented")

    @staticmethod
    def high_bounds(phases):
        raise Exception("abstract class: method not implemented")

    @classmethod
    def generate_trace(cls, model_waveform, template_params, return_wave=False):
        """

        Inputs:
        model_waveform: a Waveform object. The actual waveform data is ignored, but the start/end times and sampling rate are used to constrain the template generation.

        """

        nm = get_noise_model(model_waveform)

        srate = model_waveform['srate']
        st = model_waveform['stime']
        et = model_waveform['etime']
        npts = model_waveform['npts']

        data = np.ones((npts,)) * nm.c

            v = vals[i, :]
            arr_time = v[0]
            start = (arr_time - st) * srate
            start_idx = int(np.floor(start))
            if start_idx >= npts:
                continue

            offset = start - start_idx
            phase_env = cls.abstract_logenv_raw(v, idx_offset=offset, srate=srate)
            end_idx = start_idx + len(phase_env)
            if end_idx <= 0:
                continue

            try:
                early = max(0, -start_idx)
                overshoot = max(0, end_idx - len(data))
                data[start_idx + early:end_idx - overshoot] += np.exp(phase_env[early:len(phase_env) - overshoot])
            except Exception as e:
                print e
                raise

        if return_wave:
            wave = Waveform(data=data, segment_stats=model_waveform.segment_stats.copy(), my_stats=model_waveform.my_stats.copy())
            try:
                del wave.segment_stats['evid']
                del wave.segment_stats['event_arrivals']
            except KeyError:
                pass

            return wave
        else:
            return data

