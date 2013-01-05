from django.conf.urls import patterns, url
from coda_fits.views import FitListView, FitDetailView, FitImageView, rate_fit
from django.views.generic import DetailView, ListView
from coda_fits.models import SigvisaCodaFit, SigvisaCodaFittingRun

urlpatterns = patterns('',
    url(r'^fits$', FitListView.as_view(), name='fits'),
    url(r'^fits/runs/$',   ListView.as_view(queryset=SigvisaCodaFittingRun.objects.order_by('run_name'),
                                           context_object_name='run_list',
                                           template_name='coda_fits/runs.html')),
#    url(r'^fits/runs/(?P<pk>\d+)/$',   FitListView.as_view(queryset=SigvisaCodaFit.objects.filter('runid=pk'))),
    url(r'^fits/runs/(?P<runid>\d+)/(?P<pk>\d+)$',   FitDetailView.as_view(), name="fit_run_detail"),
    url(r'^fits/runs/(?P<runid>\d+)/(?P<fitid>\d+)/rate/$',  rate_fit),
#    url(r'^fits/(?P<pk>\d+)/$',
#        FitDetailView.as_view(),
#        name='detail'),
    url(r'^fits/runs/(?P<runid>\d+)/(?P<fitid>\d+)/visual.png$',
        FitImageView,
        name='visual'),
)
