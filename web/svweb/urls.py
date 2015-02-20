from django.conf.urls import patterns, url
from svweb.views import *
from svweb.wiggle_views import *
from svweb.model_views import *
from svweb.gridsearch_views import *
from svweb.mcmc_views import *
from svweb.event_views import *
from svweb.site_views import *
from svweb.wave_views import *
from svweb.nm_views import *
from django.views.generic import DetailView, ListView
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from svweb.models import SigvisaCodaFit, SigvisaCodaFittingRun

urlpatterns = patterns('',
                       url(r'^$', main_view, name='main'),
                       url(r'^runs/$', ListView.as_view(queryset=SigvisaCodaFittingRun.objects.order_by('run_name'),
                                                        context_object_name='run_list',
                                                        template_name='svweb/runs.html'), name="all_runs"),
                       url(r'^runs/(?P<runid>\d+)/$',
                           lambda request, runid: HttpResponseRedirect(reverse('fit_list', args=(runid,))),
                           name='fit_list'),
                       url(r'^runs/(?P<runid>\d+)/delete$', delete_run, name='delete_run'),
                       url(r'^runs/(?P<runid>\d+)/fit_quality.png$', fit_cost_quality, name='fit_cost_quality'),
                       url(r'^runs/(?P<runid>\d+)/fit_list$', fit_list_view, name='fit_list'),
                       url(r'^runs/(?P<runid>\d+)/heatmap.png$', data_heatmap_plot, name="heatmap"),
                       url(r'^runs/(?P<runid>\d+)/distance_decay.png$', data_distance_plot, name="distance_decay"),
                       url(r'^runs/(?P<runid>\d+)/pairwise.png$', data_pairwise_plot, name="pairwise_params"),
                       url(r'^runs/(?P<runid>\d+)/histogram.png$', data_histogram_plot, name="param_histogram"),
                       url(r'^fits/(?P<fitid>\d+)/detail$', fit_detail, name="fit_run_detail"),
                       url(r'^fits/(?P<fitid>\d+)/rate/$', rate_fit, name="rate_fit"),
                       url(r'^fits/(?P<fitid>\d+).png$', FitImageView, name='visual'),
                       url(r'^fits/(?P<fitid>\d+)/delete$', delete_fit, name='delete_fit'),
                       url(r'^fits/(?P<fitid>\d+)/debug$', template_debug_view, name="template_debug"),
                       url(r'^fits/(?P<fitid>\d+)/custom_phases.png$', custom_template_view, name="custom_phase_image"),
                       url(r'^fits/(?P<fitid>\d+)/template_residual.png$', template_residual_view, name="template_residual"),
                       url(r'^phases/(?P<fpid>\d+)/wiggle.png$', FitWiggleView, name='fit_wiggle_view'),
                       url(r'^phases/(?P<fpid>\d+)/$', wiggle_detail_view, name='fit_phase_wiggles'),
                       url(r'^models/$', model_list_view, name='model_list'),
                       url(r'^models/(?P<modelid>\d+)/density.png$', model_density, name='model_density'),
                       url(r'^models/(?P<modelid>\d+)/distance_plot.png$', model_distance_plot, name='model_distance_plot'),
                       url(r'^models/(?P<modelid>\d+)/mb_plot.png$', model_mb_plot, name='model_mb_plot'),
                       url(r'^models/(?P<modelid>\d+)/heatmap.png$', model_heatmap, name='model_heatmap'),
                       url(r'^models/(?P<modelid>\d+)/heatmap_std.png$', model_heatmap_std, name='model_heatmap_std'),
                       url(r'^mcmc/$', mcmc_list_view, name='mcmc_runs'),
                       url(r'^mcmc/(?P<dirname>.*?)/analyze$', mcmcrun_analyze, name='mcmcrun_analyze'),
                       url(r'^mcmc/(?P<dirname>[^/]*?)/$', mcmc_run_detail, name='mcmcrun_detail'),
                       url(r'^mcmc/(?P<dirname>.*?)/browse/(?P<path>.*)$', mcmcrun_browsedir, name='mcmcrun_browsedir'),
                       url(r'^mcmc/(?P<dirname>.*?)/waves/(?P<wn_label>.*)/vis.png$', mcmc_wave_posterior, name='mcmc_wave_posterior'),
                       url(r'^mcmc/(?P<dirname>.*?)/waves/(?P<wn_label>.*)/(?P<key1>.*?).png$', mcmc_signal_posterior_wave, name='mcmc_signal_posterior_wave'),
                       url(r'^mcmc/(?P<dirname>.*?)/waves/(?P<wn_label>.*)/arrivals_(?P<step>\d+).txt$', mcmc_arrivals, name='mcmc_arrivals'),
                       url(r'^mcmc/(?P<dirname>.*?)/params/(?P<node_label>.*).txt$', mcmc_param_posterior, name='mcmc_param_posterior'),
                       url(r'^mcmc/(?P<dirname>.*?)/event_posterior.png$', mcmc_event_posterior, name='mcmc_event_posterior'),
                       url(r'^gridsearch/$', gridsearch_list_view, name='gridsearch_list'),
                       url(r'^gridsearch/(?P<gsid>\d+)/$', gridsearch_detail_view, name='gsrun_detail'),
                       url(r'^gridsearch/(?P<gsid>\d+)/heatmap.png$', gs_heatmap_view, name='gs_heatmap'),
                       url(r'^gridsearch/(?P<gsid>\d+)/delete$', delete_gsrun, name='gsrun_delete'),
                       url(r'^gsdebug/(?P<gswid>\d+)/pickled_wave.png$', gs_pickled_env_view, name='gs_pickled_wave'),
                       url(r'^gsdebug/(?P<gswid>\d+)/phase_frame$', gs_wave_phase_frame_view, name='gs_phase_frame'),
                       url(r'^event/(?P<evid>\d+)/$', event_view, name='event'),
                       url(r'^event/(?P<evid>\d+)/context.png$', event_context_img_view, name='event_context_img'),
                       url(r'^event/(?P<evid>\d+)/wave_view$', event_wave_view, name='event_wave'),
                       url(r'^event/$', regional_event_view, name='event_region'),
                       url(r'^event/region.png$', regional_event_image_view, name='event_region_image'),
                       url(r'^site/(?P<sta>\w+)/$', site_view, name='site'),
                       url(r'^site/(?P<sta>\w+)/context.png$', site_det_img_view, name='site_det_img'),
                       url(r'^waves/$', WaveSelectView, name='wave_select'),
                       url(r'^waves/wave.png$',
                           WaveImageView,
                           name='wave_image'),
                       url(r'^nm/$', nm_list_view, name='nm_list'),
                       url(r'^nm/(?P<nmid>\d+)/detail$', nm_detail_view, name='nm_detail'),
                       url(r'^nm/(?P<nmid>\d+)/params.png$', nm_param_plot, name='nm_param_plot'),
                       url(r'^nm/(?P<nmid>\d+)/sample.png$', nm_sample, name='nm_sample'),
                       url(r'^nm/(?P<nmid>\d+)/spectrum.png$', nm_spectrum, name='nm_spectrum'),
                       url(r'^nm/(?P<nmid>\d+)/crossval.png$', nm_crossval, name='nm_crossval'),
                       )
