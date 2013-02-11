import sigvisa.database.db
from sigvisa.database.dataset import *
import time
import learn
import sys
import numpy as np

from collections import defaultdict

from optparse import OptionParser

from sigvisa.signals.io import fetch_waveform


def main():

    parser = OptionParser()

    parser.add_option("-s", "--sta", dest="sta", default=None, type="str", help="station")
    parser.add_option("-c", "--chan", dest="chan", default=None, type="str", help="channel")
    parser.add_option("-f", "--filter_str", dest="filter_str", default="", type="str", help="filter_str")
    parser.add_option("--st", dest="start_time", default=None, type="float", help="")
    parser.add_option("--et", dest="end_time", default=None, type="float", help="")

    (options, args) = parser.parse_args()

    wave = fetch_waveform(sta=sta, chan=chan, options.start_time)
