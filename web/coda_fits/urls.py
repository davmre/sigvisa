from django.conf.urls import patterns, url
from coda_fits.views import *
from coda_fits.wiggle_views import *
from django.views.generic import DetailView, ListView
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from coda_fits.models import SigvisaCodaFit, SigvisaCodaFittingRun

urlpatterns = patterns('',
    url(r'^fits/runs/$',   ListView.as_view(queryset=SigvisaCodaFittingRun.objects.order_by('run_name'),
                                           context_object_name='run_list',
                                           template_name='coda_fits/runs.html'), name="all_runs"),
    url(r'^fits/runs/(?P<runid>\d+)/$',
        lambda request, runid: HttpResponseRedirect(reverse('fit_list', args=(runid, "all", "all", "all", "all",))),
        name='fit_list_redirect'),
    url(r'^fits/runs/(?P<runid>\d+)/delete$', delete_run, name='delete_run'),
    url(r'^fits/runs/(?P<runid>\d+)/fit_quality.png$', fit_cost_quality, name='fit_cost_quality'),
    url(r'^fits/runs/(?P<runid>\d+)/filter/(?P<sta>\w+)/(?P<chan>\w+)/(?P<band>[.\w]+)/(?P<fit_quality>\w+)/$',   FitListView.as_view(), name='fit_list'),
    url(r'^fits/runs/(?P<runid>\d+)/filter/(?P<sta>\w+)/(?P<chan>\w+)/(?P<band>[.\w]+)/(?P<fit_quality>\w+)/(?P<pageid>\d+)$',   fit_detail, name="fit_run_detail"),
    url(r'^fits/runs/(?P<runid>\d+)/filter/(?P<sta>\w+)/(?P<chan>\w+)/(?P<band>[.\w]+)/(?P<fit_quality>\w+)/(?P<pageid>\d+)/rate/$',  rate_fit, name="rate_fit"),
    url(r'^fits/runs/(?P<runid>\d+)/filter/(?P<sta>\w+)/(?P<chan>\w+)/(?P<band>[.\w]+)/(?P<fit_quality>\w+)/distance_decay.png$', distance_decay, name="distance_decay"),
    url(r'^fits/(?P<fitid>\d+)/delete$', delete_fit, name='delete_fit'),
#    url(r'^fits/(?P<pk>\d+)/$',
#        FitDetailView.as_view(),
#        name='detail'),
    url(r'^fits/(?P<fitid>\d+).png$', FitImageView, name='visual'),
    url(r'^phases/(?P<fpid>\d+)/$', wiggle_detail_view, name='fit_phase_wiggles'),
    url(r'^phases/(?P<fpid>\d+)/raw_wiggle.png$', raw_wiggle_view, name='raw_wiggle'),
    url(r'^wiggles/(?P<wiggleid>\d+)/reconstructed_wiggle.png$', reconstructed_wiggle_view, name='reconstructed_wiggle'),
    url(r'^phases/(?P<fpid>\d+)/template_wiggle.png$', template_wiggle_view, name='template_wiggle'),
    url(r'^wiggles/(?P<wiggleid>\d+)/reconstructed_template_wiggle.png$', reconstructed_template_wiggle_view, name='reconstructed_template_wiggle'),
)
