from random import randrange, uniform

from lib import *

FILE_ID = 'single_dataset'
NUMBER_OF_AGENTS = 1
MAX_SAMPLES_PER_AGENT = 100
MIN_SAMPLES_PER_AGENT = 50
MAX_SIZE_OF_SAMPLE = 10
MIN_SIZE_OF_SAMPLE = 3
MU_RANGE = (1.0, 2.0)
SIGMA_RANGE = (0.1, 1.0)

sd_pool = []


def run_script():
    df = pd.DataFrame(columns=['Agent', 'sample_id', 'sample_size', 'sample', 'mu', 'sd'])

    for agent_id in range(NUMBER_OF_AGENTS):
        num_of_samples = randrange(MIN_SAMPLES_PER_AGENT, MAX_SAMPLES_PER_AGENT)
        for sample_id in range(num_of_samples):
            sample_size = randrange(MIN_SIZE_OF_SAMPLE, MAX_SIZE_OF_SAMPLE)
            curr_mu = uniform(MU_RANGE[0], MU_RANGE[1])
            curr_sd = _get_sd()
            curr_sample = np.random.normal(curr_mu, curr_sd, sample_size).tolist()
            # _save_sample_in_histogram(curr_sample, curr_sd, curr_mu)
            row = [agent_id, sample_id, sample_size, curr_sample, curr_mu, curr_sd]
            df.loc[(agent_id+1)*(sample_id+1)] = row

    df.to_csv(f'{FILE_ID}.csv')
    _write_metadata()


def _get_sd():
    if len(sd_pool) < 3:
        sd = uniform(SIGMA_RANGE[0], SIGMA_RANGE[1])
        sd_pool.append(sd)
        return sd
    return sd_pool[randrange(0, 2)]


def _save_sample_in_histogram(sample, sigma, mu):
    count, bins, ignored = plt.hist(sample, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.savefig(f'figs/his({mu},{sigma}.png', bbox_inches='tight')


def _write_metadata():
    metadata = [
        f'FILE_ID: {FILE_ID}',
        f'NUMBER_OF_AGENTS: {NUMBER_OF_AGENTS}',
        f'MAX_SAMPLES_PER_AGENT: {MAX_SAMPLES_PER_AGENT}',
        f'MIN_SAMPLES_PER_AGENT: {MIN_SAMPLES_PER_AGENT}',
        f'MAX_SIZE_OF_SAMPLE: {MAX_SIZE_OF_SAMPLE}',
        f'MIN_SIZE_OF_SAMPLE: {MIN_SIZE_OF_SAMPLE}',
        f'MU_RANGE: {MU_RANGE}',
        f'SIGMA_RANGE: {SIGMA_RANGE}',
    ]
    with open(f'{FILE_ID}.txt', 'w') as f:
        for line in metadata:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    run_script()
    print("done.")
