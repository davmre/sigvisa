from signals.template_models.paired_exp import *

def load_template_model(template_shape, **kwargs):

    if template_shape == "paired_exp":
        return PairedExpTemplateModel(**kwargs)
