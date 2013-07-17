from sigvisa.models.templates.paired_exp import PairedExpTemplateGenerator
#from sigvisa.models.templates.new_exp import NewExpTemplateGenerator

def load_template_generator(template_shape, **kwargs):
    if template_shape == "paired_exp":
        return PairedExpTemplateGenerator(**kwargs)
    else:
        raise KeyError("unknown template shape %s" % template_shape)
