# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#     * Rearrange models' order
#     * Make sure each model has one field with primary_key=True
# Feel free to rename the models, but don't rename db_table values or field names.
#
# Also note: You'll have to insert the output of 'django-admin.py sqlcustom [appname]'
# into your database.

from django.db import models
from coda_fits.fields import UnixTimestampField, BlobField

class view_options(models.Model):
    id = models.IntegerField(primary_key=True)
    smoothing = models.IntegerField()
    logscale = models.BooleanField()
    sample = models.BooleanField()

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

class SigvisaCodaFittingRun(models.Model):
    runid = models.IntegerField(primary_key=True)
    run_name = models.CharField(max_length=765, blank=True)
    iter = models.IntegerField(null=True, blank=True)
    class Meta:
        db_table = u'sigvisa_coda_fitting_run'

    def __unicode__(self):
        return "%s_iter%04d" % (self.run_name, self.runid)

class SigvisaCodaFit(models.Model):
    fitid = models.IntegerField(primary_key=True)
    runid = models.ForeignKey(SigvisaCodaFittingRun, db_column='runid')
    evid = models.IntegerField()
    sta = models.CharField(max_length=30)
    chan = models.CharField(max_length=30)
    band = models.CharField(max_length=45)
    hz = models.FloatField(null=True, blank=True)
    optim_method = models.CharField(max_length=45, blank=True)
    iid = models.IntegerField(null=True, blank=True)
    stime = UnixTimestampField(null=True, blank=True)
    etime = UnixTimestampField(null=True, blank=True)
    acost = models.FloatField(null=True, blank=True)
    dist = models.FloatField(null=True, blank=True)
    azi = models.FloatField(null=True, blank=True)
    timestamp = UnixTimestampField(null=True, blank=True)
    elapsed = models.FloatField(null=True, blank=True)
    human_approved = models.IntegerField(default=0)
    class Meta:
        db_table = u'sigvisa_coda_fit'

class SigvisaCodaFitPhase(models.Model):
    fpid = models.IntegerField(primary_key=True)
    fitid = models.ForeignKey(SigvisaCodaFit, db_column='fitid')
    phase = models.CharField(max_length=60)
    template_model = models.CharField(max_length=60, blank=True)
    param1 = models.FloatField(null=True, blank=True)
    param2 = models.FloatField(null=True, blank=True)
    param3 = models.FloatField(null=True, blank=True)
    param4 = models.FloatField(null=True, blank=True)
    wiggle_fname = models.CharField(max_length=255, blank=True)
    class Meta:
        db_table = u'sigvisa_coda_fit_phase'


class SigvisaWiggle(models.Model):
    wiggleid = models.BigIntegerField(primary_key=True)
    fpid = models.ForeignKey(SigvisaCodaFitPhase, db_column='fpid')
    stime = models.FloatField()
    etime = models.FloatField()
    srate = models.FloatField()
    timestamp = models.FloatField()
    type = models.CharField(max_length=31)
    log = models.IntegerField()
    meta0 = models.FloatField(null=True, blank=True)
    meta1 = models.FloatField(null=True, blank=True)
    meta2 = models.FloatField(null=True, blank=True)
    meta3 = models.FloatField(null=True, blank=True)
    meta_str = models.CharField(max_length=255, blank=True)
    params = BlobField() # This field type is a guess.
    class Meta:
        db_table = u'sigvisa_wiggle'


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

