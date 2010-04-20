# draws an ROC curve for SEL3 events using the scores assigned to true and
# false events
# Usage: sel3roc.py <label> <score-file-name>
# The format of a score file is:
# <orid> <1 or 0: true event or not> <score>
# ...

import sys
import matplotlib.pyplot as plt

def draw_roc(label, fname):
  scores = []
  total_true, total_false = 0, 0
  fp = open(fname)
  for line in fp:
    orid, istrue, score = line.rstrip().split()
    
    istrue = bool(int(istrue))
    score = float(score)
    
    scores.append((score, istrue))
    
    if istrue:
      total_true += 1
    else:
      total_false += 1
  fp.close()
  
  scores.sort(reverse=True)
  
  # compute the ROC curve
  x_pts, y_pts = [], []
  num_true, num_false = 0, 0
  
  for cnt, (score, istrue) in enumerate(scores):
    if istrue:
      num_true += 1
    else:
      num_false += 1
    
    if cnt % 30 == 0 or cnt == len(scores)-1:
      y_pts.append(float(num_true) / total_true)
      x_pts.append(float(num_false) / total_false)
  
  plt.plot(x_pts, y_pts, label=label)

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
  plt.xlabel("false events")
  plt.ylabel("true events")
  plt.legend(loc = "lower right")
  plt.grid(True)
  
  plt.show()
  
if __name__ == "__main__":
  main()
  
