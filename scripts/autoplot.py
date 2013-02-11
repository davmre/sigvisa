import numpy as np
import matplotlib
from optparse import OptionParser
import os


def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()

    if "DISPLAY" in os.environ:
        matplotlib.use("TkAgg")
        interactive = True
        print "interactive mode..."
    else:
        matplotlib.use("Agg")
        interactive = False
        print "non-interactive mode..."
    import matplotlib.pylab as plt

    for wave_file_name in args:
        x = np.loadtxt(wave_file_name)
        plt.figure()
        plt.plot(x)
        plt.title(wave_file_name)

        if not interactive:
            save_fname = os.path.splitext(wave_file_name)[0] + '.png'
            print "saving to ", save_fname
            plt.savefig(save_fname)
        else:
            print "plotting", wave_file_name

    print "done."
    if interactive:
        plt.show()

if __name__ == "__main__":
    main()
