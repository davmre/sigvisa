from django.conf.urls import patterns, url
from svweb.views import *
from svweb.wiggle_views import *
from svweb.model_views import *
from svweb.gridsearch_views import *
from svweb.event_views import *
from svweb.site_views import *
from svweb.wave_views import *
from django.views.generic import DetailView, ListView
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from svweb.models import SigvisaCodaFit, SigvisaCodaFittingRun

urlpatterns = patterns('',
                       url(r'^/$', main_view, name='main'),
                       url(r'^runs/$', ListView.as_view(queryset=SigvisaCodaFittingRun.objects.order_by('run_name'),
                                                        context_object_name='run_list',
                                                        template_name='svweb/runs.html'), name="all_runs"),
                       url(r'^runs/(?P<runid>\d+)/$',
                           lambda request, runid: HttpResponseRedirect(reverse('fit_list', args=(runid,))),
                           name='fit_list'),
                       url(r'^runs/(?P<runid>\d+)/delete$', delete_run, name='delete_run'),
                       url(r'^runs/(?P<runid>\d+)/fit_quality.png$', fit_cost_quality, name='fit_cost_quality'),
                       url(r'^runs/(?P<runid>\d+)/fit_list$', fit_list_view, name='fit_list'),
                       url(r'^runs/(?P<runid>\d+)/distance_decay.png$', data_distance_plot, name="distance_decay"),
                       url(r'^runs/(?P<runid>\d+)/histogram.png$', data_histogram_plot, name="param_histogram"),
                       url(r'^fits/(?P<fitid>\d+)/detail$', fit_detail, name="fit_run_detail"),
                       url(r'^fits/(?P<fitid>\d+)/rate/$', rate_fit, name="rate_fit"),

                       url(r'^fits/(?P<fitid>\d+)/delete$', delete_fit, name='delete_fit'),
                       #    url(r'^(?P<pk>\d+)/$',
                       #        FitDetailView.as_view(),
                       #        name='detail'),
                       url(r'^fits/(?P<fitid>\d+).png$', FitImageView, name='visual'),
                       url(r'^phases/(?P<fpid>\d+)/$', wiggle_detail_view, name='fit_phase_wiggles'),
                       url(r'^phases/(?P<fpid>\d+)/raw_wiggle.png$', raw_wiggle_view, name='raw_wiggle'),
                       url(r'^wiggles/(?P<wiggleid>\d+)/reconstructed_wiggle.png$',
                           reconstructed_wiggle_view, name='reconstructed_wiggle'),
                       url(r'^phases/(?P<fpid>\d+)/template_wiggle.png$', template_wiggle_view, name='template_wiggle'),
                       url(r'^wiggles/(?P<wiggleid>\d+)/reconstructed_template_wiggle.png$', reconstructed_template_wiggle_view,
                           name='reconstructed_template_wiggle'),
                       url(r'^models/$', model_list_view, name='model_list'),
                       url(r'^models/(?P<modelid>\d+)/density.png$', model_density, name='model_density'),
                       url(r'^models/(?P<modelid>\d+)/distance_plot.png$', model_distance_plot, name='model_distance_plot'),
                       url(r'^models/(?P<modelid>\d+)/heatmap.png$', model_heatmap, name='model_heatmap'),
                       url(r'^gridsearch/$', gridsearch_list_view, name='gridsearch_list'),
                       url(r'^gridsearch/(?P<gsid>\d+)/$', gridsearch_detail_view, name='gsrun_detail'),
                       url(r'^gridsearch/(?P<gsid>\d+)/heatmap.png$', gs_heatmap_view, name='gs_heatmap'),
                       url(r'^gridsearch/(?P<gsid>\d+)/delete$', delete_gsrun, name='gsrun_delete'),
                       url(r'^gsdebug/(?P<gswid>\d+)/$', gs_debug_view, name='gs_debug'),
                       url(r'^gsdebug/(?P<gswid>\d+)/overlay.png$', gs_debug_wave_view, name='gs_debug_wave'),
                       url(r'^event/(?P<evid>\d+)/$', event_view, name='event'),
                       url(r'^event/(?P<evid>\d+)/context.png$', event_context_img_view, name='event_context_img'),
                       url(r'^site/(?P<sta>\w+)/$', site_view, name='site'),
                       url(r'^site/(?P<sta>\w+)/context.png$', site_det_img_view, name='site_det_img'),
                       url(r'^waves/$', WaveSelectView, name='wave_select'),
                       url(r'^waves/wave.png$',
                           WaveImageView,
                           name='wave_image'),

                       )
