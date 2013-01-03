from django.conf.urls import patterns, url
from coda_fits.views import FitListView, FitDetailView, FitImageView
from coda_fits.models import SigvisaCodaFit

urlpatterns = patterns('',
    url(r'^fits$', FitListView.as_view(), name='fits'),
    url(r'^fits/(?P<pk>\d+)/$',
        FitDetailView.as_view(),
        name='detail'),
    url(r'^fits/(?P<fitid>\d+)/visual.png$',
        FitImageView,
        name='visual'),
#    url(r'^(?P<poll_id>\d+)/vote/$', 'polls.views.vote', name='vote'),
)
