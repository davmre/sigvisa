import numpy as np
import matplotlib.pyplot as plt
from sigvisa.plotting.plot import bounds_without_outliers
import scipy.stats

def plot_density(data, axes, title=None, draw_stats=True):

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
        htext = "mean: %.4f\nstd:  %.4f\n\nmin:  %.4f\n5%%:   %.4f\n25%%:  %.4f\n50%%:  %.4f\n75%%:  %.4f\n95%%:  %.4f\nmax:  %.4f" % (
            mean, std, np.min(data), p5, p25, median, p75, p95, np.max(data))
        axes.text(0, 1, htext, horizontalalignment='left', verticalalignment='top', transform=axes.transAxes)


def plot_histogram(data, axes=None, draw_stats=True, n_bins=None,  **kwargs):
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

    if n_bins is None:
        # freedman / diaconis rule
        bin_size = 2 * iqr / float(n) ** (1.0 / 3.0)
        n_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_size))
    axes.hist(x=data, bins=n_bins, **kwargs)

    if draw_stats:
        htext = "mean: %.4f\nstd:  %.4f\n\nmin:  %.4f\n5%%:   %.4f\n25%%:  %.4f\n50%%:  %.4f\n75%%:  %.4f\n95%%:  %.4f\nmax:  %.4f" % (
            mean, std, np.min(data), p5, p25, median, p75, p95, np.max(data))
        axes.text(0, 1, htext, horizontalalignment='left', verticalalignment='top', transform=axes.transAxes)

    return htext

# d = x = 10 + 20*np.random.randn(10000)
# plot_histogram(d)
# plt.savefig('h.png')
