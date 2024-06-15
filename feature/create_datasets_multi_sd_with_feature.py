from datetime import datetime
from random import randrange, uniform, randint

from lib import *

FILE_ID = 'feature_dataset5'
FILE_PATH_TEMPLATE = "./datasets/{}.csv"
NUMBER_OF_AGENTS = 100
NUMBER_OF_DIST_PER_AGENT = 3
MAX_SIZE_OF_SAMPLE = 210
MIN_SIZE_OF_SAMPLE = 190
MU_RANGE = (20.1, 20.8)
SIGMA_RANGE = (0.01, 0.1)
SHOULD_ROUND = True
N_DIGIT = 5


def run_script():
    df = pd.DataFrame(columns=['Agent', 'sample_size', 'sample', 'timestamps', 'mu', 'sd'])

    for agent_id in range(NUMBER_OF_AGENTS):
        sd_pool = _generate_sds(NUMBER_OF_DIST_PER_AGENT)
        mu_pool = _generate_mus(NUMBER_OF_DIST_PER_AGENT)
        for dist_num in range(NUMBER_OF_DIST_PER_AGENT):
            sample_size = randrange(MIN_SIZE_OF_SAMPLE, MAX_SIZE_OF_SAMPLE)
            curr_mu = mu_pool[dist_num]
            curr_sd = sd_pool[dist_num]
            curr_sample = _pos_normal(curr_mu, curr_sd, sample_size)
            curr_timestamps = _create_timestamps_seq(len(curr_sample))
            row = [agent_id, sample_size, curr_sample, curr_timestamps, curr_mu, curr_sd]
            df.loc[len(df.index)] = row
        # print(f"done Agent {agent_id}")

    _write_csv(df)
    _write_metadata()


def _create_timestamps_seq(size):
    start_time = datetime(2024, 1, 1, 12, 0, 0).timestamp()
    end_time = datetime(2024, 1, 1, 12, 0, 15).timestamp()
    timestamp_seq = [datetime.fromtimestamp(np.random.uniform(start_time, end_time)).timestamp() for _ in range(size)]
    timestamp_seq.sort()
    return timestamp_seq


def _pos_normal(mu, sd, sample_size):
    # make sure all numbers are in (0.02, 2.0) range, and not negative.
    # todo: fix negative possibility, maybe shift values by shifting mu
    sample = np.random.normal(mu, sd, sample_size)
    for _ in range(10):
        if not all(0.02 < x < 2.0 for x in sample):
            sample = np.random.normal(mu, sd, sample_size)
    return [_round_if_configured(x) for x in sample]


def _generate_sds(num_of_sds):
    sd_pool = []
    while len(sd_pool) < num_of_sds:
        sd = _round_if_configured(uniform(SIGMA_RANGE[0], SIGMA_RANGE[1]))
        if sd not in sd_pool:
            sd_pool.append(sd)
    return sd_pool


def _generate_mus(num_of_sds):
    mu_pool = []
    while len(mu_pool) < num_of_sds:
        sd = _round_if_configured(uniform(MU_RANGE[0], MU_RANGE[1]))
        if sd not in mu_pool:
            mu_pool.append(sd)
    return mu_pool


def _round_if_configured(x, n_digit=N_DIGIT):
    return round(x, n_digit) if SHOULD_ROUND else x


def _split_total_samples_num_to_samples_groups(n):
    result = []
    for i in range(NUMBER_OF_DIST_PER_AGENT - 1):
        x = randint(int(n / (NUMBER_OF_DIST_PER_AGENT + 1)), int(n / (NUMBER_OF_DIST_PER_AGENT - 1)))
        n = n - x
        result.append(x)
    result.append(n)
    return result


def _save_sample_in_histogram(sample, sigma, mu):
    count, bins, ignored = plt.hist(sample, 30, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.savefig(f'figs/his({mu},{sigma}.png', bbox_inches='tight')


def _write_csv(df):
    if not os.path.exists(FILE_PATH_TEMPLATE.format(FILE_ID)):
        df.to_csv(FILE_PATH_TEMPLATE.format(FILE_ID))
    else:
        df.to_csv(FILE_PATH_TEMPLATE.format(f'{FILE_ID}_new'))


def _write_metadata():
    metadata = [
        f'FILE_ID: {FILE_ID}',
        f'NUMBER_OF_AGENTS: {NUMBER_OF_AGENTS}',
        f'NUMBER_OF_DIST_PER_AGENT: {NUMBER_OF_DIST_PER_AGENT}',
        f'MAX_SIZE_OF_SAMPLE: {MAX_SIZE_OF_SAMPLE}',
        f'MIN_SIZE_OF_SAMPLE: {MIN_SIZE_OF_SAMPLE}',
        f'MU_RANGE: {MU_RANGE}',
        f'SIGMA_RANGE: {SIGMA_RANGE}',
        f'SHOULD_ROUND: {SHOULD_ROUND}',
        f'N_DIGIT: {N_DIGIT}',
    ]
    with open(f'./datasets/metadata/{FILE_ID}.txt', 'w') as f:
        for line in metadata:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    run_script()
    print("done.")
