from lib import *

NUMBER_OF_AGENTS = 1
MAX_SAMPLES_PER_AGENT = 100
MIN_SAMPLES_PER_AGENT = 50
MAX_SIZE_OF_SAMPLE = 10
MIN_SIZE_OF_SAMPLE = 3
MU_RANGE = (1.0, 2.0)
SIGMA_RANGE = (0.1, 1.0)

sd_pool = []


def run_script(file_id='single_dataset'):
    file_id = f'{file_id}_{int(datetime.now().timestamp())}'
    print(f'file id: {file_id}')

    df = pd.DataFrame(columns=['Agent', 'sample_id', 'sample_size', 'sample', 'mu', 'sd'])

    for agent_id in range(NUMBER_OF_AGENTS):
        num_of_samples = randrange(MIN_SAMPLES_PER_AGENT, MAX_SAMPLES_PER_AGENT)
        for sample_id in range(num_of_samples):
            sample_size = randrange(MIN_SIZE_OF_SAMPLE, MAX_SIZE_OF_SAMPLE)
            curr_mu = uniform(MU_RANGE[0], MU_RANGE[1])
            curr_sd = _get_sd()
            curr_sample = np.random.normal(curr_mu, curr_sd, sample_size).tolist()
            # save_sample_in_histogram(curr_sample, curr_sd, curr_mu)  # add import from lib.utils
            row = [agent_id, sample_id, sample_size, curr_sample, curr_mu, curr_sd]
            df.loc[(agent_id+1)*(sample_id+1)] = row

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df.to_csv(f'./datasets/{file_id}.csv')
    _write_metadata(file_id)


def _get_sd():
    if len(sd_pool) < 3:
        sd = uniform(SIGMA_RANGE[0], SIGMA_RANGE[1])
        sd_pool.append(sd)
        return sd
    return sd_pool[randrange(0, 2)]


def _write_metadata(file_id):
    metadata = [
        f'FILE_ID: {file_id}',
        f'NUMBER_OF_AGENTS: {NUMBER_OF_AGENTS}',
        f'MAX_SAMPLES_PER_AGENT: {MAX_SAMPLES_PER_AGENT}',
        f'MIN_SAMPLES_PER_AGENT: {MIN_SAMPLES_PER_AGENT}',
        f'MAX_SIZE_OF_SAMPLE: {MAX_SIZE_OF_SAMPLE}',
        f'MIN_SIZE_OF_SAMPLE: {MIN_SIZE_OF_SAMPLE}',
        f'MU_RANGE: {MU_RANGE}',
        f'SIGMA_RANGE: {SIGMA_RANGE}',
    ]
    with open(f'./datasets/metadata/{file_id}.txt', 'w') as f:
        for line in metadata:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_script(file_id=sys.argv[1])
    else:
        run_script()
    print("done.")
