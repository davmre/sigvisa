import numpy as np
import sys
import os
from sigvisa import Sigvisa


class TemplateGenerator(object):

    @staticmethod
    def params():
        raise NotImplementedError("template model does not implement params")

    @staticmethod
    def default_param_vals():
        raise NotImplementedError("template model does not implement default_param_vals")

    @staticmethod
    def model_name():
        raise NotImplementedError("template model does not implement model_name")

    @staticmethod
    def abstract_logenv_raw(vals):
        raise NotImplementedError("template model does not implement abstract_logenv_raw")

    def low_bounds(self):
        raise NotImplementedError("template model does not implement low_bounds")

    def high_bounds(self):
        raise NotImplementedError("template model does not implement high_bounds")

    def create_param_node(self, graph, **kwargs):
        return graph.setup_site_param_node(**kwargs)
