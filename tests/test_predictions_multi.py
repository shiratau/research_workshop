import ast

from lib import *
from multi import predict_multi_sd

DEVIATION_PERCENTAGE = 20
UP_LIMIT = (100 + DEVIATION_PERCENTAGE) / 100
DOWN_LIMIT = (100 - DEVIATION_PERCENTAGE) / 100


def test_predict_multi_numbers_from_seq():
    df = pd.DataFrame(
        {
            "input": pd.Series([
                np.array([3.461, 3.478, 3.478, 3.485, 3.489, 3.489, 3.492]),
                np.array([3.469, 3.481, 3.481, 3.495, 3.495]),
                np.array([3.519, 3.528, 3.542, 3.542, 3.546, 3.55]),
                np.array([3.523, 3.529, 3.543, 3.543, 3.552, 3.554, 3.555])
            ]),
            "output": pd.Series([
                np.array([3.281, 3.28, 3.27]),
                np.array([3.271, 3.199, 3.261]),
                np.array([3.247, 3.255, 3.3]),
                np.array([3.214, 3.211, 3.124])
            ])
        }
    )
    predictions, y_test = predict_multi_sd.run(df)
    print(predictions)
    print(y_test)

    for i in range(len(predictions.flat)):
        prd = predictions.flat[i]
        exp = y_test.flat[i]
        cmp = _compare_result_in_allowed_range(prd, exp)
        print(f'prediction: {prd}, expected: {exp}, comparison: {cmp}')


def test_predict_multi_dataset77():
    df = pd.DataFrame(
        {
            "input": pd.Series([
                np.array([0.486, 0.677, 0.453, 0.547, 0.527, 0.408, 0.505, 0.745, 0.78, 0.636, 0.647, 0.819, 0.351, 0.381, 0.383, 0.309, 0.333]),
                np.array([0.336, 0.319, 0.342, 0.277, 0.332, 0.337, 0.286, 0.305, 0.21, 0.232, 0.216, 0.247, 0.206, 0.271, 0.233, 0.31, 0.263, 0.292, 0.279, 0.233]),
                np.array([0.648, 0.642, 0.481, 0.755, 0.52, 0.685, 0.618, 0.713, 0.47, 0.56, 0.57, 0.8, 0.725, 0.711, 0.099, 0.095, 0.114, 0.088, 0.085, 0.128, 0.16]),
                np.array([0.209, 0.203, 0.243, 0.228, 0.286, 0.318, 0.24, 0.24, 0.261, 0.259, 0.255, 0.278, 0.726, 0.718, 0.749, 0.72, 0.682, 0.688])
            ]),
            "output": pd.Series([
                np.array([0.085, 0.089, 0.021]),
                np.array([0.024, 0.041, 0.02]),
                np.array([0.093, 0.091, 0.048]),
                np.array([0.047, 0.021, 0.023])
            ])
        }
    )
    predictions, y_test = predict_multi_sd.run(df)
    print(predictions)
    print(y_test)

    for i in range(len(predictions.flat)):
        prd = predictions.flat[i]
        exp = y_test.flat[i]
        cmp = _compare_result_in_allowed_range(prd, exp)
        print(f'prediction: {prd}, expected: {exp}, comparison: {cmp}')


def test_predict_sd_normal():
    file_id = "feature_dataset5"
    # raw_df = pd.read_csv(f'./../multi/datasets/{file_id}.csv')
    raw_df = pd.read_csv(f'./../feature/datasets/{file_id}.csv')
    print(raw_df.head())
    samples_input, sd_output = _get_data_from_df(raw_df)
    df = pd.DataFrame(
        {
            "input": pd.Series(samples_input),
            "output": pd.Series(sd_output)
        }
    )

    predictions, y_test = predict_multi_sd.run(df)
    print(predictions)
    print(y_test)

    df = pd.DataFrame(columns=['expected', 'prediction', 'compare', 'binary'])

    suc = 0
    for i in range(len(predictions)):
        sd_prd_set = predictions[i]
        sd_exp_set = y_test[i]
        for j in range(len(sd_prd_set)):
            prd = sd_prd_set[j]
            exp = sd_exp_set[j]
            cmp = _compare_result_in_allowed_range(prd, exp)
            print(f'set: {i}, prediction: {prd}, expected: {exp}, comparison: {cmp}')
            if cmp:
                suc += 1

            df.loc[len(df.index)] = [exp, prd, cmp, int(cmp)]

    print(f"num of success: {suc}")
    print(f"num of fails: {len(predictions)*3 - suc}")
    _export_result(df, file_id)


def _get_data_from_df(raw_df):
    num_of_agents = max(raw_df['Agent']) + 1
    print(f'number of agents: {num_of_agents}')
    samples_input = []
    sd_output = []

    for agent_id in range(num_of_agents):
        agents_rows = raw_df.loc[raw_df['Agent'] == agent_id]
        samples_bulk_feed = []
        sd_series = []

        for sample in agents_rows["sample"]:
            samples_bulk_feed += ast.literal_eval(sample)

        samples_input.append(pd.Series(np.array(samples_bulk_feed, dtype=float)))

        for sd in agents_rows["sd"]:
            if sd not in sd_series:
                sd_series.append(sd)

        sd_output.append(pd.Series(np.array(sd_series, dtype=float)))

    print(samples_input)
    print(sd_output)
    return samples_input, sd_output


def _compare_result_in_allowed_range(prediction, y_test_value) -> bool:
    return y_test_value * DOWN_LIMIT <= prediction <= y_test_value * UP_LIMIT


def _export_result(df, file_id):
    path = "./results/multi/result_multi_{}.csv"

    if not os.path.exists(path.format(file_id)):
        df.to_csv(path.format(file_id))
    else:
        df.to_csv(path.format(f'{file_id}_new'))
