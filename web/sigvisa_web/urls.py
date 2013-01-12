from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'sigvisa.views.home', name='home'),
    # url(r'^sigvisa/', include('sigvisa.foo.urls')),

                       url(r'^sigvisa/', include('coda_fits.urls')),
                       url(r'^sigvisa/', include('waves.urls')),


    # Uncomment the admin/doc line below to enable admin documentation:
    url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', include(admin.site.urls)),
)