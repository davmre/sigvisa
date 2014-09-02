from sigvisa.models.templates.paired_exp import PairedExpTemplateGenerator
from sigvisa.models.templates.lin_polyexp import LinPolyExpTemplateGenerator
#from sigvisa.models.templates.new_exp import NewExpTemplateGenerator

def load_template_generator(template_shape, **kwargs):
    if template_shape == "paired_exp":
        return PairedExpTemplateGenerator(**kwargs)
    if template_shape == "lin_polyexp":
        return LinPolyExpTemplateGenerator(**kwargs)

    else:
        raise KeyError("unknown template shape %s" % template_shape)
