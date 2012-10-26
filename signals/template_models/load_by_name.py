from signals.template_models.paired_exp import *

def load_template_model(template_shape, run_name, model_type):

    if template_shape == "paired_exp":
        return PairedExponentialTemplateModel(run_name=run_name, model_type=model_type)

