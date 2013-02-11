import numpy as np
import sys
import os

from sigvisa import *
from train_coda_models import analyze_model_fname


def main():

    # call using the form
    # python learn/cross_validation_aggregate_results.py eval/run8/iter_00/paired_exp/*/*/P/BHZ/freq_2.0_3.0/*_results.txt
    result_files = sys.argv[1:]

    # nested dict structure: target->model_type->key->sta->value
    results = NestedDict()
    stations = set()

    for result_file in result_files:
        cv_dir, fname = os.path.split(result_file)
        model_type = "_".join(fname.split("_")[:-1])
        print model_type
        d = analyze_model_fname(os.path.join(cv_dir, "dummyhash." + model_type))
        f = open(result_file, 'r')

        stations.add(d['sta'])

        for line in f:
            key, value = line.split()
            value = float(value)
            results[d['target'] + "_" + key][model_type][d['sta']] = value

    stations = list(stations)
    for target_key in results.keys():
        fname_aggregate = "eval/%s_aggregate_results.txt" % target_key
        f = open(fname_aggregate, 'w')
        print "writing to", fname_aggregate

        f.write("model_type\t" + "\t".join(stations) + "\n")
        for model_type in sorted(results[target_key].keys()):
            f.write("%s\t" % model_type)
            for sta in stations:
                try:
                    f.write("%.4f\t" % results[target_key][model_type][sta])
                except TypeError:
                    print "no results for", target_key, model_type, sta
                    f.write("\t")
            f.write("\n")
        f.close()

if __name__ == "__main__":
    main()
