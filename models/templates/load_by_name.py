from sigvisa.models.templates.paired_exp import *

def load_template_model(template_shape, **kwargs):

    if template_shape == "paired_exp":
        return PairedExpTemplateModel(**kwargs)
