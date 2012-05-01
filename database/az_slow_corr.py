# Copyright (c) 2012, Bayesian Logic, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bayesian Logic, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# Bayesian Logic, Inc. BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# 
import os
from math import sin, cos, asin, atan2, degrees, radians, hypot, sqrt

SASC_WARNINGS = 1

class AzSlowCorr:
  def __init__(self, filename):
    """
    def_sx_corr;      /* Default x-vector slowness (sx) corr */
    def_sy_corr;      /* Default y-vector slowness (sy) corr */
    def_slow_mdl_err; /* Default slowness modeling error */
    a11;              /* A11 x-vector affine slowness coefficient */
    a12;              /* A12 y-vector affine slowness coefficient */
    a21;              /* A21 x-vector affine slowness coefficient */
    a22;              /* A22 y-vector affine slowness coefficient */
    bins array of tuples:
      lb_slo, ub_slo, lb_az, ub_az, corr_slo, corr_az, mdl_err_slow, mdl_err_az

    """
    self.bins = []
    
    READ_DEFAULTS, READ_NUM_BINS, READ_CORRECTIONS = range(3)
    state = READ_DEFAULTS
    
    fp = open(filename, "r")
    
    for line in fp.readlines():
      line = line.rstrip()              # remove trailing newlines

      if not(len(line)) or line[0]=='#':
        continue                        # skip comments      

      if state == READ_DEFAULTS:
        def_vals = [float(x) for x in line.split()]

        if len(def_vals) == 3:
          self.def_sx_corr, self.def_sy_corr, self.def_slow_mdl_err = def_vals
          self.a11, self.a12, self.a21, self.a22 = 1., 0., 0., 1.
        elif len(def_vals) == 7:
          (self.def_sx_corr, self.def_sy_corr, self.def_slow_mdl_err,
           self.a11, self.a12, self.a21, self.a22) = def_vals
        else:
          raise ValueError("Invalid defaults in SASC file" + filename)

        state = READ_NUM_BINS
        
      elif state == READ_NUM_BINS:
        num_bins = int(line)
        state = READ_CORRECTIONS

      elif state == READ_CORRECTIONS:
        #print "'" + line + "'"
        self.bins.append(tuple(float(x) for x in line.split()))
        # there better be exactly 8 numbers in the line
        if len(self.bins[-1]) != 8:
          raise ValueErorr("Bad bin number %d in SASC file %s" %
                           (len(self.bins)), filename)

    # on exit check if we read any corrections, and if we did, whether we
    # read the right number of corrections
    if state == READ_DEFAULTS:
      raise ValueError("Empty SASC file: " + filename)
    
    elif state == READ_NUM_BINS:
      num_bins = 0

    elif state == READ_CORRECTIONS:
      if num_bins != len(self.bins):
        global SASC_WARNINGS
        if SASC_WARNINGS == 0:
          print ("Warning: Expecting %d corrections got %d in SASC file: %s"
                 % (num_bins, len(self.bins), filename))
          SASC_WARNINGS = 1

    fp.close()
  
  def correct(self, raw_azimuth, raw_slow, delaz, delslo):
    """
    Returns corrected azimuth slowness and their errors
    """
    # if the input values are not valid return them unmodified
    if raw_azimuth < 0 or raw_slow <= 0 or delaz < 0 or delslo < 0:
      return raw_azimuth, raw_slow, delaz, delslo
    
    azimuth, slow = raw_azimuth, raw_slow
    tot_az_error, tot_slow_err = delaz, delslo
    
    # first convert default vector slowness corrections to a localized
    # slowness/azimuth correction.  Also store slowness modeling error.
    slow_mdl_err = self.def_slow_mdl_err
    
    # Determine default modeling error for azimuth as a function of
    # the default slowness modeling error.
    azimuth_mdl_err = slow_mdl_err / (2.0 * slow)
    if azimuth_mdl_err < 1.0:
      azimuth_mdl_err = degrees(2.0 * asin(azimuth_mdl_err))
    else:
      azimuth_mdl_err = 180.0
    
    for bin in self.bins:
      (lb_slo, ub_slo, lb_az, ub_az, corr_slo, corr_az,
       mdl_err_slow, mdl_err_az) = bin
      
      if (slow < ub_slo and slow >= lb_slo and azimuth < ub_az and
          azimuth >= lb_az):
        azimuth_mdl_err = mdl_err_az
        slow_mdl_err = mdl_err_slow

        # Update azimuth and slowness with corrected values. 
        azimuth -= corr_az
        slow -= corr_slo
        if azimuth < 0.0:
          azimuth += 360.0
        if azimuth > 360.0:
          azimuth -= 360.0
        
        break
        
    # Apply affine and default slowness vector corrections here.  First 
    # decompose the original azimuth and slowness into vector slowness 
    # componenets (sx, sy), and then, apply affine transform.  Then 
    # apply default slowness vector corrections in x- and y-directions.
    # Finally, adjust input azimuth and slow based on these updated
    # slowness vector component adjustments.

    # Affine transform: corrected_sx = (a11*sx + a12*sy) - def_sx_corr
    #                   corrected_sy = (a21*sx + a22*sy) - def_sy_corr
    # where,
    #      sx and sy have already been bin corrected.
    # 

    if slow > 0.0:
      azr = radians(azimuth)
      sx = slow * sin (azr)
      sy = slow * cos (azr)
      adj_sx = self.a11 * sx + self.a12 * sy
      adj_sy = self.a21 * sx + self.a22 * sy
      sx = adj_sx
      sy = adj_sy
      
      # Apply default sx and sy corrections here to get the total affine
      # transformed slowness vector positiions.
      sx -= self.def_sx_corr            # Apply default sx corr. here 
      sy -= self.def_sy_corr            # Apply default sy corr. here
      
      # Revert back to azimuth/slowness space
      azimuth = degrees(atan2 (sx, sy))
      if azimuth < 0.0:
        azimuth += 360.0
      slow = hypot (sx, sy)
      
    # Total azimuth and slowness errors are an RMS measure of the
    # combined measurement error (delaz and delslo) and the modeling
    # error (azimuth_mdl_err and slow_mdl_err).

    if azimuth_mdl_err < 180.0:
      tot_az_err = hypot(delaz, azimuth_mdl_err)
    else:
      tot_az_err = 180.0

    tot_slow_err = hypot(delslo, slow_mdl_err)

    return azimuth, slow, tot_az_error, tot_slow_err
          
def load_az_slow_corr(sasc_dir):
  """
  Returns a dictionary of station names mapped to AzSlowCorr structures
  """
  corr_dict = {}
  for fname in os.listdir(sasc_dir):
    if fname[:5] == 'sasc.':
      sta = fname[5:]
      corr_dict[sta] = AzSlowCorr(os.path.join(sasc_dir, fname))
      
  return corr_dict

if __name__ == "__main__":
  corr_dict = load_az_slow_corr(os.path.join(os.path.curdir, '..',
                                             'parameters', 'sasc'))
  print len(corr_dict), "corrections loaded"

  t = ['ASAR', 86.808, 8.219, 1.698, 0.244]
  
  print t, "->", corr_dict[t[0]].correct(*t[1:])
