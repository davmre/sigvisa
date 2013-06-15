from sigvisa.models.templates.paired_exp import *

def load_template_generator(template_shape, **kwargs):
    if template_shape == "paired_exp":
        return PairedExpTemplateGenerator(**kwargs)
