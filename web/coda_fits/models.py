# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#     * Rearrange models' order
#     * Make sure each model has one field with primary_key=True
# Feel free to rename the models, but don't rename db_table values or field names.
#
# Also note: You'll have to insert the output of 'django-admin.py sqlcustom [appname]'
# into your database.

from django.db import models

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

class IdcxArrivalNet(models.Model):
    sta = models.CharField(max_length=18)
    time = models.FloatField()
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
        db_table = u'idcx_arrival_net'

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
        db_table = u'sigvisa_coda_fitting_runs'

    def __unicode__(self):
        return "%s_%d" % (self.run_name, self.iter)

class SigvisaCodaFit(models.Model):
    fitid = models.IntegerField(primary_key=True)
    runid = models.ForeignKey(SigvisaCodaFittingRun, db_column='runid')
    evid = models.IntegerField()
    sta = models.CharField(max_length=30)
    chan = models.CharField(max_length=30)
    lowband = models.FloatField()
    highband = models.FloatField()
    phase = models.CharField(max_length=60)
    atime = models.FloatField(null=True, blank=True)
    peak_delay = models.FloatField(null=True, blank=True)
    coda_height = models.FloatField(null=True, blank=True)
    coda_decay = models.FloatField(null=True, blank=True)
    optim_method = models.CharField(max_length=45, blank=True)
    iid = models.IntegerField(null=True, blank=True)
    stime = models.FloatField(null=True, blank=True)
    etime = models.FloatField(null=True, blank=True)
    acost = models.FloatField(null=True, blank=True)
    dist = models.FloatField(null=True, blank=True)
    azi = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = u'sigvisa_coda_fits'


class SigvisaWiggleWfdisc(models.Model):
    wiggleid = models.IntegerField(primary_key=True)
    runid = models.ForeignKey(SigvisaCodaFittingRun, db_column='runid')
    arid = models.IntegerField()
    siteid = models.IntegerField(null=True, blank=True)
    phaseid = models.IntegerField(null=True, blank=True)
    band = models.CharField(max_length=30)
    chan = models.CharField(max_length=30)
    evid = models.IntegerField(null=True, blank=True)
    fname = models.CharField(max_length=765, blank=True)
    snr = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = u'sigvisa_wiggle_wfdisc'

class StaticPhaseid(models.Model):
    id = models.IntegerField(primary_key=True)
    phase = models.CharField(max_length=60, unique=True, blank=True)
    timedef = models.CharField(max_length=3, blank=True)
    class Meta:
        db_table = u'static_phaseid'

class StaticSite(models.Model):
    sta = models.CharField(max_length=18, primary_key=True)
    ondate = models.IntegerField(primary_key=True)
    offdate = models.IntegerField(null=True, blank=True)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    elev = models.FloatField(null=True, blank=True)
    staname = models.CharField(max_length=150, blank=True)
    statype = models.CharField(max_length=12, blank=True)
    refsta = models.CharField(max_length=18, blank=True)
    dnorth = models.FloatField(null=True, blank=True)
    deast = models.FloatField(null=True, blank=True)
    lddate = models.DateTimeField(null=True, blank=True)
    class Meta:
        db_table = u'static_site'

class StaticSitechan(models.Model):
    sta = models.CharField(max_length=18, primary_key=True)
    chan = models.CharField(max_length=24, primary_key=True)
    ondate = models.IntegerField(primary_key=True)
    chanid = models.IntegerField(null=True, blank=True)
    offdate = models.IntegerField(null=True, blank=True)
    ctype = models.CharField(max_length=12, blank=True)
    edepth = models.FloatField(null=True, blank=True)
    hang = models.FloatField(null=True, blank=True)
    vang = models.FloatField(null=True, blank=True)
    descrip = models.CharField(max_length=150, blank=True)
    lddate = models.DateTimeField(null=True, blank=True)
    class Meta:
        db_table = u'static_sitechan'

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

