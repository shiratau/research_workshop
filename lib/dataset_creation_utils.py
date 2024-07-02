from . import *


def save_sample_in_histogram(sample, sigma, mu):
    count, bins, ignored = plt.hist(sample, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.savefig(f'figs/his({mu},{sigma}.png', bbox_inches='tight')
