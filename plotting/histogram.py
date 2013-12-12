import numpy as np
import matplotlib.pyplot as plt
from sigvisa.plotting.plot import bounds_without_outliers
import scipy.stats

def plot_density(data, axes, title=None, draw_stats=True, true_value=None):

    if np.std(data) > 0.0:
        density = scipy.stats.kde.gaussian_kde(data)
        minb, maxb = bounds_without_outliers(data)
        x = np.linspace(minb, maxb, 100)
        axes.plot(x, density(x))
    else:
        axes.plot(data[0], 1.0)

    if title is not None:
        axes.set_title(title)

    if draw_stats:
        data = sorted(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        median = np.median(data)
        p5 = data[n / 20]
        p25 = data[n / 4]
        p75 = data[n * 3 / 4]
        p95 = data[n * 19 / 20]
        iqr = p75 - p25

        htext = "mean: %.4f\nstd:  %.4f\n" % (mean, std)

        if true_value is not None:
            htext += "true: %.4f\n" % true_value

        htext += "\nmin:  %.4f\n5%%:   %.4f\n25%%:  %.4f\n50%%:  %.4f\n75%%:  %.4f\n95%%:  %.4f\nmax:  %.4f" % (np.min(data), p5, p25, median, p75, p95, np.max(data))
        axes.text(0, 1, htext, horizontalalignment='left', verticalalignment='top', transform=axes.transAxes)


def plot_histogram(data, axes=None, title=None, draw_stats=True, n_bins=None, bin_size=None, bins=None, true_value=None, trueval_label=None, return_full=False,  **kwargs):
    data = sorted(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    p5 = data[n / 20]
    p25 = data[n / 4]
    p75 = data[n * 3 / 4]
    p95 = data[n * 19 / 20]
    iqr = p75 - p25

    if axes is None:
        fig = plt.Figure()
        axes = fig.add_subplot(111)

    if title is not None:
        axes.set_title(title)


    if n_bins is None:
        if bin_size is None:
            # freedman / diaconis rule
            if iqr > 0:
                bin_size = 2 * iqr / float(n) ** (1.0 / 3.0)
            else:
                bin_size = 2 * std / float(n) ** (1.0 / 3.0)

        n_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_size))
    n, bins, patches = axes.hist(x=data, bins=n_bins, **kwargs)


    if draw_stats:
        htext = "mean: %.4f\nstd:  %.4f\n" % (mean, std)

        if true_value is not None:
            htext += "true: %.4f\n" % true_value

        htext += "\nmin:  %.4f\n5%%:   %.4f\n25%%:  %.4f\n50%%:  %.4f\n75%%:  %.4f\n95%%:  %.4f\nmax:  %.4f" % (np.min(data), p5, p25, median, p75, p95, np.max(data))
        axes.text(0, 1, htext, horizontalalignment='left', verticalalignment='top', transform=axes.transAxes)
    else:
        htext=None

    if true_value is not None:
        ymin, ymax = axes.get_ylim()
        axes.bar(left=[true_value,], height=ymax-ymin,
                 width= bin_size/2.0, bottom=ymin, edgecolor="red", color="red", linewidth=1, alpha=.5)
        if trueval_label is not None:
            axes.text(true_value + bin_size, ymax - (ymax-ymin)*.1, trueval_label, color="red", fontsize=10)

    if return_full:
        return htext, bins, patches
    else:
        return htext

def plot_gaussian_fit(data, axes):
    mean = np.mean(data)
    std = np.std(data)
    xs = np.linspace(np.min(data), np.max(data), 100)
    ys = 1.0/(std * np.sqrt(2*np.pi)) * np.exp(-.5 * ((xs-mean)/std)**2)
    if not normalize:
        ys *= len(data)
    axes.plot(xs, ys)
# d = x = 10 + 20*np.random.randn(10000)
# plot_histogram(d)
# plt.savefig('h.png')
