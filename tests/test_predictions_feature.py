import ast

from lib import *
from feature import predict_featured_sd
from .utils import compare_result_in_allowed_range, export_result, parse_result_for_multi_sd_dataset, set_and_get_wd

TEST_FILE_ID = "feature_dataset5"


def test_predict_multi_numbers_with_timestamp_from_seq():
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
    predictions, y_test = predict_featured_sd.run(df)
    print(predictions)
    print(y_test)

    for i in range(len(predictions.flat)):
        prd = predictions.flat[i]
        exp = y_test.flat[i]
        cmp = compare_result_in_allowed_range(prd, exp)
        print(f'prediction: {prd}, expected: {exp}, comparison: {cmp}')


def test_predict_from_dataset():
    this_dir = set_and_get_wd()
    raw_df = pd.read_csv(os.path.join(this_dir, f'..\\feature\\datasets', f'{TEST_FILE_ID}.csv'))
    print(raw_df.head())

    # preparing data for model:
    samples_input, timestamps_input, sd_output = _get_data_from_df(raw_df)
    df = pd.DataFrame(
        {
            "input_samples": pd.Series(samples_input),
            "input_timestamps": pd.Series(timestamps_input),
            "output": pd.Series(sd_output)
        }
    )

    # run model and get predictions and expected values:
    predictions, y_test = predict_featured_sd.run(df)
    print(f"RAW PREDICTIONS:\n{predictions}\n\nRAW EXPECTED:{y_test}\n")

    result_df = parse_result_for_multi_sd_dataset(predictions, y_test)
    export_result(result_df, "feature", TEST_FILE_ID)


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
