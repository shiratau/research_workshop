import ast

from lib import *
from multi import predict_multi_sd

DEVIATION_PERCENTAGE = 30
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


def test_predict_sd_normal():
    raw_df = pd.read_csv("multi_dataset7.csv")
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

    print(f"num of success: {suc}")
    print(f"num of fails: {len(predictions)*3 - suc}")


def _get_data_from_df(raw_df):
    num_of_agents = 10  # todo: read from dataset metadata
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
