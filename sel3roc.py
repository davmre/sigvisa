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
# draws an ROC curve for SEL3 events using the scores assigned to true and
# false events
# Usage: sel3roc.py <label> <score-file-name>
# The format of a score file is:
# <orid> <1 or 0: true event or not> <score>
# ...

import sys
import matplotlib.pyplot as plt

curr_color = 0
colors=["red", "blue", "orange", "yellow", "brown", "purple"]

NUMPTS = 200

def draw_roc(label, fname):
  fp = open(fname)
  false_sel3, true_sel3 = [], []
  for line in fp:
    orid, istrue, score = line.rstrip().split()
    score = float(score)
    istrue = bool(int(istrue))
    
    if istrue:
      true_sel3.append(score)
    else:
      false_sel3.append(score)
  
  fp.close()

  min_score = min(min(true_sel3), min(false_sel3)) - 1
  max_score = max(max(true_sel3), max(false_sel3)) + 1

  # compute the ROC curve
  x_pts, y_pts = [], []

  for cnt in range(NUMPTS):
    sep = min_score + float(cnt) * (max_score - min_score) / NUMPTS
    y = float(len(filter(lambda x: x>sep, true_sel3))) / len(true_sel3)
    x = float(len(filter(lambda x: x>sep, false_sel3))) / len(false_sel3)
      
    x_pts.append(x)
    y_pts.append(y)
    
  global curr_color
  plt.plot(x_pts, y_pts, label=label, color=colors[curr_color])
  curr_color += 1

def main():
  if (len(sys.argv)-1) % 2 != 0 or sys.argv < 3:
    print "Usage: sel3roc.py <label> <score-file-name> "\
          "[<label> <score-file>].."
    sys.exit(1)
  
  plt.figure()
  plt.title("ROC curve for SEL3 events")

  for i in range(1, len(sys.argv), 2):
    label = sys.argv[i]
    fname = sys.argv[i+1]
  
    draw_roc(label, fname)

  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.xlabel("False positive rate")
  plt.ylabel("True positive rate")
  plt.legend(loc = "lower right")
  plt.grid(True)
  
  plt.show()
  
if __name__ == "__main__":
  main()
  
