from django.contrib import admin
from coda_fits.models import SigvisaCodaFit, SigvisaCodaFitPhase, SigvisaCodaFittingRun


class PhaseInline(admin.TabularInline):
    model = SigvisaCodaFitPhase
    extra=1

class FitAdmin(admin.ModelAdmin):
    fields = ['runid', 'evid', 'sta', 'chan', 'band', 'optim_method', 'iid', 'stime', 'etime', 'acost', 'dist', 'azi', 'human_approved']
    list_display = ('runid', 'evid', 'sta', 'chan', 'band', 'stime', 'human_approved')
    list_filter = ('runid', 'sta', 'chan', 'band', 'stime', 'human_approved')
    inlines = [PhaseInline]
    # date_hierarchy = 'stime'

admin.site.register(SigvisaCodaFittingRun)
admin.site.register(SigvisaCodaFit, FitAdmin)
