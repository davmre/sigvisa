#
# Slowness-azimuth station correction table
#
# INK, 20000106 00:20:18
#
# time period: 1996-01-01 - 1999-12-16
# 6789 input slowness-azimuth readings
# 357 bins populated with observations
# 1871 readings contributed to the corrections
# provisional default corrections:  sx: 0.727 , sy: 0.433
# azimuthal variance of default corrections: 61.866
# no global trend has been detected
# overall sdev of slowness residuals: 1.933
# overall sdev of  azimuth residuals: 21.081
# uncorrected bins, slowness residual sdev: 2.158
# uncorrected bins,  azimuth residual sdev: 24.275
# corrected bins, slowness residual sdev: 0.819
# corrected bins,  azimuth residual sdev: 5.573
#
# The corrections are determined for the bins on a circular grid
# defined by the slowness and azimuth intervals: [Slo,Shi),[Alo,Ahi).
# Grid definition file: /home/jinwang/NewSASC/grid/SASC.new.grid
#
# Only those corrections (i.e. median of observed slowness-azimuth residuals
# in a bin) are listed where the criteria for corrections were satisfied:
#    ( number of observations >= 10 && 
#      slowness residual std error <= 1.00 &&
#      azimuth residual std error <= 15.00 )
#
# A global trend is detected if at least 7.0% of bins are populated with
# observations and the azimuthal variance of corrections is less than 33.0.
# If a global trend exists in the corrections, a default correction
# is given for all detections. Further fine-tuning bin corrections are 
# applied to the detections in the bins defined in this file.
# In the lack of global trend the default corrections are set to zero and
# the default modeling error calculated only from the unbinned observations.
#
# Structure of the slowness-azimuth correction (SASC) table:
#    comment lines begin with '#'
#    line 1: Contains default slowness vector corrections (sx,sy) and the
#            slowness modeling error (azimuth modeling error is derived)
#       column 1: default slowness correction, x-component
#       column 2: default slowness correction, y-component
#       column 3: default slowness modeling error (sec/deg)
#    line 2: number of bins with correction (ncor)
#            (if only defaults are given the rest of this file remains empty)
#    line 3 - line ncor+2: bin corrections and modeling errors
#       column 1: bottom of slowness interval
#       column 2: top of slowness interval
#       column 3: bottom of azimuth interval
#       column 4: top of azimuth interval
#       column 5: slowness correction
#       column 6: azimuth correction
#       column 7: slowness modeling error
#       column 8: azimuth modeling error
#
# Default slowness vector corrections (sx,sy) and slowness modeling error:
   0.000    0.000    2.158
# Number of user-specified SASC bins:
29
# User-specified bin definitions, S-A corrections and S-A modeling errors:
   4.000    6.000  105.000  120.000    0.501   -7.578    0.922   13.197
   4.000    6.000  225.000  240.000    0.233  -17.568    0.921   11.458
   6.000    8.000    0.000   10.000    1.398   23.628    0.515    4.699
   6.000    8.000  110.000  120.000    1.208   -7.699    0.804    4.507
   6.000    8.000  130.000  140.000    0.427   -0.497    0.628   11.800
   6.000    8.000  140.000  150.000    0.416    3.738    0.913   10.838
   6.000    8.000  280.000  290.000   -1.004    6.695    0.945    4.134
   6.000    8.000  290.000  300.000   -0.943   14.245    0.838    3.781
   6.000    8.000  300.000  310.000   -0.749   22.439    0.748    5.392
   6.000    8.000  330.000  340.000    0.912   14.702    0.704   10.423
   6.000    8.000  340.000  350.000    1.000    7.809    0.660   11.716
   6.000    8.000  350.000  360.000    1.061   17.309    0.570    9.259
   8.000   10.000   45.000   52.500    1.036   -6.618    0.723    3.694
   8.000   10.000   52.500   60.000    1.719   -1.680    0.900    8.521
   8.000   10.000  120.000  127.500    2.226   -5.515    0.754    6.943
   8.000   10.000  127.500  135.000    1.909    0.919    0.442    3.921
   8.000   10.000  142.500  150.000    1.478   -0.004    0.907   12.089
   8.000   10.000  285.000  292.500    0.603    9.745    0.927    3.801
   8.000   10.000  292.500  300.000    0.708   17.445    0.980    4.925
   8.000   10.000  345.000  352.500    2.013   10.495    0.862   12.608
  10.000   12.000    0.000    6.000    1.997    7.854    0.564    4.065
  10.000   12.000    6.000   12.000    2.339   13.916    0.636    3.419
  10.000   12.000  240.000  246.000    1.149   -3.612    0.800    5.647
  10.000   12.000  270.000  276.000    2.971   -1.674    0.916    7.287
  10.000   12.000  276.000  282.000    2.683    4.528    0.880    5.420
  10.000   12.000  294.000  300.000    3.106   17.518    0.973    4.262
  10.000   12.000  354.000  360.000    2.140    3.596    0.567    4.338
  16.000   18.000  225.000  230.000    3.162   -7.299    0.759    4.768
  16.000   18.000  230.000  235.000    3.581   -3.252    0.537    2.580
