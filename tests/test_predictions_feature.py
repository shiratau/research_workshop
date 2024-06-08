import ast

from lib import *
from feature import predict_multi_featured_sd_new

DEVIATION_PERCENTAGE = 20
UP_LIMIT = (100 + DEVIATION_PERCENTAGE) / 100
DOWN_LIMIT = (100 - DEVIATION_PERCENTAGE) / 100


def test_predict_multi_numbers_from_seq():
    df = pd.DataFrame(
        {
            "input_samples": pd.Series([
                np.array([3.461, 3.478, 3.478, 3.485, 3.489, 3.489, 3.492]),
                np.array([3.469, 3.481, 3.481, 3.495, 3.495]),
                np.array([3.519, 3.528, 3.542, 3.542, 3.546, 3.55]),
                np.array([3.523, 3.529, 3.543, 3.543, 3.552, 3.554, 3.555])
            ]),
            "input_timestamps": pd.Series([
                np.array([1704103213.513568, 1704103213.51402, 1704103213.561272, 1704103213.59531, 1704103213.778183, 1704103213.844442, 1704103213.894931]),
                np.array([1704103200.026567, 1704103200.046179, 1704103200.19055, 1704103200.341469, 1704103200.445509]),
                np.array([1704103203.46775, 1704103203.639603, 1704103203.737883, 1704103203.824579, 1704103203.861546, 1704103204.0398]),
                np.array([1704103206.749316, 1704103206.802049, 1704103206.863207, 1704103206.8859, 1704103206.952539, 1704103207.015557, 1704103207.175337])
            ]),
            "output": pd.Series([
                np.array([3.281, 3.28, 3.27]),
                np.array([3.271, 3.199, 3.261]),
                np.array([3.247, 3.255, 3.3]),
                np.array([3.214, 3.211, 3.124])
            ])
        }
    )
    predictions, y_test = predict_multi_featured_sd_new.run(df)
    print(predictions)
    print(y_test)

    for i in range(len(predictions.flat)):
        prd = predictions.flat[i]
        exp = y_test.flat[i]
        cmp = _compare_result_in_allowed_range(prd, exp)
        print(f'prediction: {prd}, expected: {exp}, comparison: {cmp}')


def test_feature():
    file_id = "feature_dataset2"
    raw_df = pd.read_csv(f'./../feature/datasets/{file_id}.csv')
    samples_input, timestamps_input, sd_output = _get_data_from_df(raw_df)
    df = pd.DataFrame(
        {
            "input_samples": pd.Series(samples_input),
            "input_timestamps": pd.Series(timestamps_input),
            "output": pd.Series(sd_output)
        }
    )

    # features = tf.ragged.constant(timestamps_input, dtype=tf.float64)
    # labels = tf.ragged.constant(samples_input, dtype=tf.float64)
    # dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    #
    # df = pd.DataFrame(
    #     {
    #         "input": dataset,
    #         "output": pd.Series(sd_output)
    #     }
    # )

    predictions, y_test = predict_multi_featured_sd_new.run(df)

    print(predictions)
    print(y_test)

    df = pd.DataFrame(columns=['expected', 'prediction', 'compare'])

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

            df.loc[len(df.index)] = [exp, prd, cmp]

    print(f"num of success: {suc}")
    print(f"num of fails: {len(predictions)*3 - suc}")
    _export_result(df, file_id)
    print(raw_df.head())


def _get_data_from_df(raw_df):
    num_of_agents = max(raw_df["Agent"]) + 1
    print(f'number of agents: {num_of_agents}')
    samples_input = []
    timestamps_input = []
    sd_output = []

    for agent_id in range(num_of_agents):
        agents_rows = raw_df.loc[raw_df["Agent"] == agent_id]
        samples_bulk_feed = []
        sd_series = []
        timestamps_seq_feed = []

        for sample in agents_rows["sample"]:
            samples_bulk_feed += ast.literal_eval(sample)

        samples_input.append(pd.Series(np.array(samples_bulk_feed, dtype=float)))

        for timestamps_seq in agents_rows["timestamps"]:
            timestamps_seq_feed += ast.literal_eval(timestamps_seq)

        timestamps_input.append(pd.Series(np.array(timestamps_seq_feed, dtype=float)))

        for sd in agents_rows["sd"]:
            if sd not in sd_series:
                sd_series.append(sd)

        sd_output.append(pd.Series(np.array(sd_series, dtype=float)))

    print(samples_input)
    print(timestamps_input)
    print(sd_output)
    return samples_input, timestamps_input, sd_output


def _compare_result_in_allowed_range(prediction, y_test_value) -> bool:
    return y_test_value * DOWN_LIMIT <= prediction <= y_test_value * UP_LIMIT


def _export_result(df, file_id):
    path = "./results/{}.csv"

    if not os.path.exists(path.format(file_id)):
        df.to_csv(path.format(file_id))
    else:
        df.to_csv(path.format(f'{file_id}_new'))
