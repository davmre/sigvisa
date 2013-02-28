import numpy as np
from sigvisa.models.noise.noise_util import get_noise_model

def get_wiggle_param_model_ids(runid, sta, chan, band, phase, model_type, basisid):
    s = Sigvisa()
    cursor = s.dbconn.cursor()

    sql_query = "select wpmid from sigvisa_template_param_model where model_type = '%s' and sta='%s' and chan='%s' and band='%s' and phase='%s' and basisid=%d and fitting_runid=%d" % (model_type, sta, chan, band, phase, basisid, runid)
    cursor.execute(sql_query)
    wpmids = cursor.fetchall()
    return wpmids

def WiggleModelNode(ClusterNode):

    def __init__(self, label="", parents={}, children=[], wiggle_param_model_ids=[], atime_offset_seconds=0.5):

        super(self, ClusterNode).__init__(label=label, nodes=[], parents=parents, children=children)

        s = Sigvisa()
        cursor = s.dbconn.cursor()

        for wpmid in wiggle_param_models_ids:
            sql_query = "select param, model_type, model_fname, wpmid, basisid from sigvisa_wiggle_param_model where wpmid=%d" % (wpmid)
            cursor.execute(sql_query)
            models.append(cursor.fetchone())
        cursor.close()

        # load all relevant models as new graph nodes
        self.nodes = []
        self.nodeDict = NestedDict()
        basedir = os.getenv("SIGVISA_HOME")
        basisid = None
        for (param, db_model_type, fname, wpmid, wpm_basisid) in sorted(models):

            model = load_model(os.path.join(basedir, fname), db_model_type)
            mNode = Node(model=model, parents=parents, children=children, label='%s' % (param))
            self.nodes.append(mNode)

            # ensure that all models are using the same basis
            assert(basisid is None or basisid == wpm_basisid)
            basisid = wpm_basisid

        self.featurizer = load_basis(basisid)
        assert(len(self.nodes) == featurizer.nparams())

        self.atime_offset_seconds=atime_offset_seconds

    def get_wiggle(self, npts):
        wiggle = self.featurizer.signal_from_features(features = self.get_value(), npts=npts)
        return wiggle

class DummyWiggleModelNode(WiggleModelNode):

    def __init__(self, label="", parents={}, children=[]):
        super(self, ClusterNode).__init__(label=label, nodes=[], parents=parents, children=children)

    def get_value(self):
        return []

    def get_wiggle(self, npts):
        return np.ones((npts,))

