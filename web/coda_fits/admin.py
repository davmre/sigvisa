from django.contrib import admin
from coda_fits.models import SigvisaCodaFit, SigvisaCodaFittingRun

class FitAdmin(admin.ModelAdmin):
    fields = ['runid', 'evid', 'sta', 'chan', 'lowband', 'highband', 'phase', 'atime', 'peak_delay', 'coda_height', 'coda_decay', 'optim_method', 'iid', 'stime', 'etime', 'acost', 'dist', 'azi']
    list_display = ('runid', 'evid', 'sta', 'chan', 'lowband', 'highband', 'phase')
    list_filter = ('runid', 'sta', 'chan', 'lowband')

admin.site.register(SigvisaCodaFittingRun)
admin.site.register(SigvisaCodaFit, FitAdmin)
