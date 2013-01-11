from django.conf.urls import patterns, url
from waves.views import WaveImageView, WaveSelectView

urlpatterns = patterns('',
    url(r'^waves/$', WaveSelectView, name='wave_select'),
    url(r'^waves/wave.png$',
        WaveImageView,
        name='wave_image'),
)
