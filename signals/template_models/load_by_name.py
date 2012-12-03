from signals.template_models.paired_exp import *

def load_template_model(template_shape, run_name, run_iter, model_type):

    if template_shape == "paired_exp":
        return PairedExpTemplateModel(run_name=run_name, run_iter=run_iter, model_type=model_type)

