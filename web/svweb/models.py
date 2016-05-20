# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#     * Rearrange models' order
#     * Make sure each model has one field with primary_key=True
# Feel free to rename the models, but don't rename db_table values or field names.
#
# Also note: You'll have to insert the output of 'django-admin.py sqlcustom [appname]'
# into your database.

from django.db import models
from svweb.fields import UnixTimestampField, BlobField

from sigvisa.models.noise.noise_model import NoiseModel
from sigvisa.signals.io import fetch_waveform

class view_options(models.Model):
    id = models.IntegerField(primary_key=True)
    smoothing = models.IntegerField()
    logscale = models.BooleanField()
    wiggle = models.BooleanField()
    noise = models.BooleanField()

    class Meta:
        db_table = u'coda_fits_view_options'


class Dataset(models.Model):
    label = models.CharField(max_length=60)
    start_time = models.FloatField()
    end_time = models.FloatField()

    class Meta:
        db_table = u'dataset'


class IdcxArrival(models.Model):
    sta = models.CharField(max_length=18)
    time = models.FloatField(primary_key=True)
    arid = models.IntegerField()
    jdate = models.IntegerField(null=True, blank=True)
    stassid = models.IntegerField(null=True, blank=True)
    chanid = models.IntegerField(null=True, blank=True)
    chan = models.CharField(max_length=24, blank=True)
    iphase = models.CharField(max_length=24, blank=True)
    stype = models.CharField(max_length=3, blank=True)
    deltim = models.FloatField(null=True, blank=True)
    azimuth = models.FloatField(null=True, blank=True)
    delaz = models.FloatField(null=True, blank=True)
    slow = models.FloatField(null=True, blank=True)
    delslo = models.FloatField(null=True, blank=True)
    ema = models.FloatField(null=True, blank=True)
    rect = models.FloatField(null=True, blank=True)
    amp = models.FloatField(null=True, blank=True)
    per = models.FloatField(null=True, blank=True)
    logat = models.FloatField(null=True, blank=True)
    clip = models.CharField(max_length=3, blank=True)
    fm = models.CharField(max_length=6, blank=True)
    snr = models.FloatField(null=True, blank=True)
    qual = models.CharField(max_length=3, blank=True)
    auth = models.CharField(max_length=45, blank=True)
    commid = models.IntegerField(null=True, blank=True)
    lddate = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = u'idcx_arrival'


class IdcxWfdisc(models.Model):
    sta = models.CharField(max_length=18)
    chan = models.CharField(max_length=24)
    time = models.FloatField()
    wfid = models.IntegerField(primary_key=True)
    chanid = models.IntegerField(null=True, blank=True)
    jdate = models.IntegerField(null=True, blank=True)
    endtime = models.FloatField(null=True, blank=True)
    nsamp = models.IntegerField(null=True, blank=True)
    samprate = models.FloatField(null=True, blank=True)
    calib = models.FloatField(null=True, blank=True)
    calper = models.FloatField(null=True, blank=True)
    instype = models.CharField(max_length=18, blank=True)
    segtype = models.CharField(max_length=3, blank=True)
    datatype = models.CharField(max_length=6, blank=True)
    clip = models.CharField(max_length=3, blank=True)
    dir = models.CharField(max_length=192, blank=True)
    dfile = models.CharField(max_length=96, blank=True)
    foff = models.IntegerField(null=True, blank=True)
    commid = models.IntegerField(null=True, blank=True)
    lddate = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = u'idcx_wfdisc'


class LebArrival(models.Model):
    sta = models.CharField(max_length=18)
    time = models.FloatField()
    arid = models.IntegerField(primary_key=True)
    jdate = models.IntegerField(null=True, blank=True)
    stassid = models.IntegerField(null=True, blank=True)
    chanid = models.IntegerField(null=True, blank=True)
    chan = models.CharField(max_length=24, blank=True)
    iphase = models.CharField(max_length=24, blank=True)
    stype = models.CharField(max_length=3, blank=True)
    deltim = models.FloatField(null=True, blank=True)
    azimuth = models.FloatField(null=True, blank=True)
    delaz = models.FloatField(null=True, blank=True)
    slow = models.FloatField(null=True, blank=True)
    delslo = models.FloatField(null=True, blank=True)
    ema = models.FloatField(null=True, blank=True)
    rect = models.FloatField(null=True, blank=True)
    amp = models.FloatField(null=True, blank=True)
    per = models.FloatField(null=True, blank=True)
    logat = models.FloatField(null=True, blank=True)
    clip = models.CharField(max_length=3, blank=True)
    fm = models.CharField(max_length=6, blank=True)
    snr = models.FloatField(null=True, blank=True)
    qual = models.CharField(max_length=3, blank=True)
    auth = models.CharField(max_length=45, blank=True)
    commid = models.IntegerField(null=True, blank=True)
    lddate = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = u'leb_arrival'


class LebAssoc(models.Model):
    arid = models.IntegerField(primary_key=True)
    orid = models.IntegerField()
    sta = models.CharField(max_length=18, blank=True)
    phase = models.CharField(max_length=24, blank=True)
    belief = models.FloatField(null=True, blank=True)
    delta = models.FloatField(null=True, blank=True)
    seaz = models.FloatField(null=True, blank=True)
    esaz = models.FloatField(null=True, blank=True)
    timeres = models.FloatField(null=True, blank=True)
    timedef = models.CharField(max_length=3, blank=True)
    azres = models.FloatField(null=True, blank=True)
    azdef = models.CharField(max_length=3, blank=True)
    slores = models.FloatField(null=True, blank=True)
    slodef = models.CharField(max_length=3, blank=True)
    emares = models.FloatField(null=True, blank=True)
    wgt = models.FloatField(null=True, blank=True)
    vmodel = models.CharField(max_length=45, blank=True)
    commid = models.IntegerField(null=True, blank=True)
    lddate = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = u'leb_assoc'


class LebOrigin(models.Model):
    lat = models.FloatField()
    lon = models.FloatField()
    depth = models.FloatField()
    time = models.FloatField()
    orid = models.IntegerField(primary_key=True)
    evid = models.IntegerField(null=True, blank=True)
    jdate = models.IntegerField(null=True, blank=True)
    nass = models.IntegerField(null=True, blank=True)
    ndef = models.IntegerField(null=True, blank=True)
    ndp = models.IntegerField(null=True, blank=True)
    grn = models.IntegerField(null=True, blank=True)
    srn = models.IntegerField(null=True, blank=True)
    etype = models.CharField(max_length=21, blank=True)
    depdp = models.FloatField(null=True, blank=True)
    dtype = models.CharField(max_length=3, blank=True)
    mb = models.FloatField(null=True, blank=True)
    mbid = models.IntegerField(null=True, blank=True)
    ms = models.FloatField(null=True, blank=True)
    msid = models.IntegerField(null=True, blank=True)
    ml = models.FloatField(null=True, blank=True)
    mlid = models.IntegerField(null=True, blank=True)
    algorithm = models.CharField(max_length=45, blank=True)
    auth = models.CharField(max_length=45, blank=True)
    commid = models.IntegerField(null=True, blank=True)
    lddate = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = u'leb_origin'

class SigvisaNoiseModel(models.Model):
    nmid = models.IntegerField(primary_key=True)
    timestamp = models.FloatField()
    sta = models.CharField(max_length=10)
    chan = models.CharField(max_length=10)
    band = models.CharField(max_length=15)
    smooth = models.IntegerField(null=True, blank=True)
    env = models.CharField(max_length=1)
    hz = models.FloatField()
    window_stime = models.FloatField()
    window_len = models.FloatField()
    model_type = models.CharField(max_length=15)
    nparams = models.IntegerField()
    mean = models.FloatField()
    std = models.FloatField()
    fname = models.CharField(max_length=255)
    created_for_hour = models.IntegerField()

    class Meta:
        db_table = u'sigvisa_noise_model'

    def load(self):
        return NoiseModel.load_from_file(self.fname, self.model_type)

    def get_data(self):
        env_filter = "env;" if self.env.startswith("t") else ""
        return fetch_waveform(str(self.sta), str(self.chan), self.window_stime, self.window_stime + self.window_len).filter('%s;%ssmooth_%d;hz_%.2f' % (self.band, env_filter, self.smooth or 0, self.hz))


class SigvisaCodaFittingRun(models.Model):
    runid = models.IntegerField(primary_key=True)
    run_name = models.CharField(max_length=765, blank=True)
    iter = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = u'sigvisa_coda_fitting_run'
        ordering = ['run_name', 'iter']

    def __unicode__(self):
        return "%s_iter%04d" % (self.run_name, self.runid)


class SigvisaCodaFit(models.Model):
    fitid = models.IntegerField(primary_key=True)
    runid = models.ForeignKey(SigvisaCodaFittingRun, db_column='runid')
    nmid = models.ForeignKey(SigvisaNoiseModel, db_column='nmid')
    evid = models.IntegerField()
    sta = models.CharField(max_length=30)
    chan = models.CharField(max_length=30)
    band = models.CharField(max_length=45)
    smooth = models.IntegerField(null=True, blank=True)
    hz = models.FloatField(null=True, blank=True)
    tmpl_optim_method = models.CharField(max_length=1024, blank=True)
    wiggle_optim_method = models.CharField(max_length=1024, blank=True)
    optim_log = models.CharField(max_length=2048, blank=True)
    iid = models.IntegerField(null=True, blank=True)
    stime = UnixTimestampField(null=True, blank=True)
    etime = UnixTimestampField(null=True, blank=True)
    acost = models.FloatField(null=True, blank=True)
    dist = models.DecimalField(null=True, blank=True, max_digits=12, decimal_places=6)
    azi = models.DecimalField(null=True, blank=True, max_digits=12, decimal_places=6)
    timestamp = UnixTimestampField(null=True, blank=True)
    elapsed = models.FloatField(null=True, blank=True)
    human_approved = models.IntegerField(default=0)
    env = models.CharField(max_length=1)

    class Meta:
        db_table = u'sigvisa_coda_fit'


class SigvisaCodaFitPhase(models.Model):
    fpid = models.IntegerField(primary_key=True)
    fitid = models.ForeignKey(SigvisaCodaFit, db_column='fitid')
    phase = models.CharField(max_length=60)
    template_model = models.CharField(max_length=60, blank=True)
    arrival_time = models.FloatField(null=True, blank=True)
    peak_offset = models.FloatField(null=True, blank=True)
    coda_height = models.FloatField(null=True, blank=True)
    coda_decay = models.FloatField(null=True, blank=True)
    peak_decay = models.FloatField(null=True, blank=True)
    amp_transfer = models.FloatField(null=True, blank=True)
    mult_wiggle_std = models.FloatField(null=True, blank=True)
    wiggle_stime = models.FloatField(null=True, blank=True)
    message_fname = models.CharField(max_length=255, blank=True)
    wiggle_family = models.CharField(max_length=20, blank=True)

    class Meta:
        db_table = u'sigvisa_coda_fit_phase'

class SigvisaTtrConsistency(models.Model):
    fpid = models.ForeignKey(SigvisaCodaFitPhase, db_column='fpid')
    ttr_residual = models.FloatField(null=True, blank=True)
    ttr_neighbor_median = models.FloatField(null=True, blank=True)
    ttr_neighbor_stddev = models.FloatField(null=True, blank=True)
    neighbor_fitids = models.CharField(max_length=512, blank=True)

    class Meta:
        db_table = u'sigvisa_ttr_consistency'


class SigvisaWiggleBasis(models.Model):
    basisid = models.IntegerField(primary_key=True)
    family_name = models.CharField(max_length=63)
    basis_type = models.CharField(max_length=31)
    srate = models.FloatField()
    logscale = models.CharField(max_length=1)
    dimension = models.IntegerField()
    npts = models.IntegerField()
    max_freq = models.FloatField(null=True, blank=True)
    training_runid = models.ForeignKey(SigvisaCodaFittingRun, db_column="training_runid", blank=True)
    training_set_fname = models.CharField(max_length=255, blank=True)
    training_sta = models.CharField(max_length=10, blank=True)
    training_chan = models.CharField(max_length=10, blank=True)
    training_band = models.CharField(max_length=15, blank=True)
    training_phase = models.CharField(max_length=10, blank=True)
    basis_fname = models.CharField(max_length=255, blank=True)

    class Meta:
        db_table = u'sigvisa_wiggle_basis'


class SigvisaWiggle(models.Model):
    wiggleid = models.IntegerField(primary_key=True)
    fpid = models.ForeignKey(SigvisaCodaFitPhase, db_column='fpid')
    basisid = models.ForeignKey(SigvisaWiggleBasis, db_column='basisid')
    timestamp = UnixTimestampField()
    params = BlobField()  # This field type is a guess.
    class Meta:
        db_table = u'sigvisa_wiggle'

class SigvisaHoughDetectionModel(models.Model):
    modelid = models.IntegerField(primary_key=True)
    fitting_runid = models.ForeignKey(SigvisaCodaFittingRun, db_column='fitting_runid')
    sta = models.CharField(max_length=10)
    phase = models.CharField(max_length=20)
    phase_context = models.CharField(max_length=40)
    model_fname = models.CharField(max_length=256)
    class Meta:
        db_table = u'sigvisa_hough_detection_model'
    
class SigvisaParamModel(models.Model):
    modelid = models.IntegerField(primary_key=True)
    fitting_runid = models.ForeignKey(SigvisaCodaFittingRun, db_column='fitting_runid')
    wiggle_basisid = models.ForeignKey(SigvisaWiggleBasis, db_column='wiggle_basisid', null=True, blank=True)
    template_shape = models.CharField(max_length=15, blank=True)
    param = models.CharField(max_length=15)
    site = models.CharField(max_length=10)
    chan = models.CharField(max_length=10)
    band = models.CharField(max_length=15)
    phase = models.CharField(max_length=10)
    max_acost = models.FloatField()
    min_amp = models.FloatField()
    require_human_approved = models.CharField(max_length=1)
    model_type = models.CharField(max_length=31)
    model_fname = models.CharField(max_length=255)
    training_set_fname = models.CharField(max_length=255)
    n_evids = models.IntegerField()
    shrinkage_iter = models.IntegerField()
    training_ll = models.FloatField()
    timestamp = UnixTimestampField()
    elapsed = models.FloatField()
    hyperparams = models.CharField(max_length=1024)
    shrinkage = models.CharField(max_length=1024)
    optim_method = models.CharField(max_length=1024)

    class Meta:
        db_table = u'sigvisa_param_model'

class SigvisaGridsearchRun(models.Model):
    gsid = models.IntegerField(primary_key=True)
    evid = models.IntegerField()
    timestamp = UnixTimestampField()
    elapsed = models.FloatField()
    lon_nw = models.FloatField()
    lat_nw = models.FloatField()
    lon_se = models.FloatField()
    lat_se = models.FloatField()
    pts_per_side = models.IntegerField()
    likelihood_method = models.CharField(max_length=63)
    optim_method = models.CharField(max_length=1024, blank=True, null=True)
    max_evtime_proposals = models.IntegerField()
    true_depth = models.CharField(max_length=1)
    true_time = models.CharField(max_length=1)
    true_mb = models.CharField(max_length=1)
    phases = models.CharField(max_length=127)
    wiggle_model_type = models.CharField(max_length=31)
    heatmap_fname = models.CharField(max_length=255)

    class Meta:
        db_table = u'sigvisa_gridsearch_run'
        ordering = ['gsid']


class SigvisaGsrunWave(models.Model):
    gswid = models.IntegerField(primary_key=True)
    gsid = models.ForeignKey(SigvisaGridsearchRun, db_column='gsid')
    nmid = models.ForeignKey(SigvisaNoiseModel, db_column='nmid')
    sta = models.CharField(max_length=10)
    chan = models.CharField(max_length=10)
    band = models.CharField(max_length=15)
    stime = models.FloatField()
    etime = models.FloatField()
    hz = models.FloatField()

    class Meta:
        db_table = u'sigvisa_gsrun_wave'
        ordering = ['gswid']


class SigvisaGsrunModel(models.Model):
    gsmid = models.IntegerField(primary_key=True)
    gswid = models.ForeignKey(SigvisaGsrunWave, db_column='gswid')
    modelid = models.ForeignKey(SigvisaParamModel, db_column='modelid')

    class Meta:
        db_table = u'sigvisa_gsrun_model'


class StaticPhaseid(models.Model):
    id = models.IntegerField(primary_key=True)
    phase = models.CharField(max_length=60, unique=True, blank=True)
    timedef = models.CharField(max_length=3, blank=True)

    class Meta:
        db_table = u'static_phaseid'


class StaticSiteid(models.Model):
    id = models.IntegerField(primary_key=True)
    sta = models.CharField(max_length=18, unique=True, blank=True)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    elev = models.FloatField(null=True, blank=True)
    staname = models.CharField(max_length=150, blank=True)
    statype = models.CharField(max_length=6, blank=True)

    class Meta:
        db_table = u'static_siteid'
