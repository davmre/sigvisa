import time
import os
import numpy as np

from sigvisa.database.signal_data import execute_and_return_id
from sigvisa.models import TimeSeriesDist


class NoiseModel(TimeSeriesDist):
    def location(self):
        raise NotImplementedError('abstract base class')

    def scale(self):
        raise NotImplementedError('abstract base class')

    def noise_model_type(self):
        raise NotImplementedError('abstract base class')

    def order(self):
        raise NotImplementedError('abstract base class')

    def save_to_db(self, dbconn, sta, chan, band, hz, window_stime, window_len, fname, hour):

        model_type = self.noise_model_type()
        nparams = self.order()

        mean = self.location()
        std = self.scale()

        sql_query = "insert into sigvisa_noise_model (sta, chan, band, hz, window_stime, window_len, model_type, nparams, mean, std, fname, created_for_hour, timestamp) values ('%s', '%s', '%s', %f, %f, %f, '%s', %d, %f, %f, '%s', %d, %f)" % (sta, chan, band, hz, window_stime, window_len, model_type, nparams, mean, std, fname, hour, time.time())
        return execute_and_return_id(dbconn, sql_query, "nmid")

    @staticmethod
    def load_from_db(dbconn, sta, chan, band, hz, order, model_type, hour, return_extra=False):

        order_cond = "and nparams=%d" % order if (model_type=='ar' and order != "auto") else ""

        cursor = dbconn.cursor()
        sql_query = "select nmid, window_stime, window_len, mean, std, fname from sigvisa_noise_model where sta='%s' and chan='%s' and band='%s' and hz=%.2f and model_type='%s' and created_for_hour=%d %s" % (sta, chan, band, hz, model_type, hour, order_cond)
        cursor.execute(sql_query)
        models = cursor.fetchall()
        cursor.close()

        assert(len(models) <= 1) # otherwise we're confused which model to load
        if len(models) == 0:
            model = None
            nm = None
        else:
            model = models[0]
            nm = NoiseModel.load_from_file(nm_fname = model[5], model_type=model_type)

        if return_extra:
            return nm, model
        else:
            return nm

    @staticmethod
    def load_from_file(nm_fname, model_type):
        full_fname = os.path.join(os.getenv('SIGVISA_HOME'), nm_fname)
        if model_type.lower() == "ar":
            from sigvisa.models.noise.armodel.model import ARModel
            model = ARModel.load_from_file(full_fname)
        elif model_type.lower() == "l1":
            from sigvisa.models.noise.iid import L1IIDModel
            model = L1IIDModel.load_from_file(full_fname)
        else:
            raise Exception("unrecognized model type %s" % (model_type))
        return model

