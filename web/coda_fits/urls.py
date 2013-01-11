from django.conf.urls import patterns, url
from coda_fits.views import FitListView, fit_detail, FitImageView, rate_fit
from django.views.generic import DetailView, ListView
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from coda_fits.models import SigvisaCodaFit, SigvisaCodaFittingRun

urlpatterns = patterns('',
    url(r'^fits/runs/$',   ListView.as_view(queryset=SigvisaCodaFittingRun.objects.order_by('run_name'),
                                           context_object_name='run_list',
                                           template_name='coda_fits/runs.html')),
    url(r'^fits/runs/(?P<runid>\d+)/$',
        lambda request, runid: HttpResponseRedirect(reverse('fit_list', args=(runid, "all", "all", "all", "all",))),
        name='fit_list_redirect'),
    url(r'^fits/runs/(?P<runid>\d+)/filter/(?P<sta>\w+)/(?P<chan>\w+)/(?P<band>\w+)/(?P<fit_quality>\w+)/$',   FitListView.as_view(), name='fit_list'),
    url(r'^fits/runs/(?P<runid>\d+)/filter/(?P<sta>\w+)/(?P<chan>\w+)/(?P<band>\w+)/(?P<fit_quality>\w+)/(?P<pageid>\d+)$',   fit_detail, name="fit_run_detail"),
    url(r'^fits/runs/(?P<runid>\d+)/filter/(?P<sta>\w+)/(?P<chan>\w+)/(?P<band>\w+)/(?P<fit_quality>\w+)/(?P<pageid>\d+)/rate/$',  rate_fit, name="rate_fit"),
#    url(r'^fits/(?P<pk>\d+)/$',
#        FitDetailView.as_view(),
#        name='detail'),
    url(r'^fits/(?P<fitid>\d+).png$',
        FitImageView,
        name='visual'),
)
