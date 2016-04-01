import django
import django.views.generic
from django.shortcuts import render_to_response, get_object_or_404
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from django.template import RequestContext
from django.http import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.core.paginator import Paginator
from django_easyfilters import FilterSet

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


import numpy as np
import os

from sigvisa.models.logistic_regression import LogisticRegressionModel


from svweb.models import SigvisaHoughDetectionModel

import cPickle as pickle
from sigvisa import Sigvisa

class ModelsFilterSet(FilterSet):
    fields = [
        'fitting_runid',
        'sta',
        'phase',
        'phase_context',
    ]


def detmodel_list_view(request):
    models = SigvisaHoughDetectionModel.objects.all()
    model_filter = ModelsFilterSet(models, request.GET)
    return render_to_response("svweb/detmodels.html",
                              {'model_list': model_filter.qs,
                               'model_filter': model_filter,
                               }, context_instance=RequestContext(request))


def plot_det_model_probs(request, modelid):

    max_dist = float(request.GET.get("max_dist", "2000"))
    min_dist = float(request.GET.get("min_dist", "0"))
    depth = float(request.GET.get("depth", "0"))

    s = Sigvisa()

    model_obj = SigvisaHoughDetectionModel.objects.get(modelid=modelid)

    sta = model_obj.sta
    phase = model_obj.phase
    phase_context = model_obj.phase_context
    title = "detection model for %s %s depth %.1f context %s" % (sta, phase, depth, phase_context)

    with open(os.path.join(s.homedir, model_obj.model_fname), 'rb') as f:
        model = pickle.load(f)

    fig = Figure(figsize=(8, 5), dpi=144)
    fig.patch.set_facecolor('white')
    axes = fig.add_subplot(111)

    mbs = [2.0, 3.0, 4.0, 5.0]
    dists = np.linspace(min_dist, max_dist, 200)
    for mb in mbs:
        detprobs = []
        for dist in dists:
            x= np.array((mb, np.exp(mb), depth, np.sqrt(depth), dist, np.log(dist)))
            detprobs.append(model.predict_prob(x))
        axes.plot(dists, detprobs, label="mb=%.1f" % mb)

    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles, labels)

    axes.set_xlim((min_dist, max_dist))
    axes.set_ylim((0, 1))
    axes.set_title(title)
    
    canvas = FigureCanvas(fig)
    response = django.http.HttpResponse(content_type='image/png')
    fig.tight_layout()
    canvas.print_png(response)
    return response

