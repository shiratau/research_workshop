import ast

from lib import *
from single import predict_single_sd
from .utils import compare_result_in_allowed_range, export_result, set_and_get_wd

TEST_FILE_ID = "single_dataset"


def test_predict_number_from_seq():
    df = pd.DataFrame(
        {
            "input": pd.Series([
                np.array([3.461, 3.478, 3.478, 3.485, 3.489, 3.489, 3.492]),
                np.array([3.469, 3.481, 3.481, 3.495, 3.495]),
                np.array([3.519, 3.528, 3.542, 3.542, 3.546, 3.55]),
                np.array([3.523, 3.529, 3.543, 3.543, 3.552, 3.554, 3.555])
            ]),
            "output": [3.281, 3.271, 3.247, 3.214]
        }
    )
    predictions, y_test = predict_single_sd.run(df)
    print(predictions)
    print(y_test)
    compare_result_in_allowed_range(predictions.flat[0], y_test.flat[0])


def test_predict_from_dataset():
    this_dir = set_and_get_wd()
    raw_df = pd.read_csv(os.path.join(this_dir, "..\\single\\datasets", f'{TEST_FILE_ID}.csv'))
    print(raw_df.head())

    # preparing data for model:
    samples_input = [np.array(ast.literal_eval(sample), dtype=float) for sample in raw_df["sample"]]
    sd_output = [_ for _ in raw_df["sd"]]

    input_df = pd.DataFrame(
        {
            "input": pd.Series(samples_input),
            "output": sd_output
        }
    )

    # run model and get predictions and expected values:
    predictions, y_test = predict_single_sd.run(input_df)
    print(f"RAW PREDICTIONS:\n{predictions}\n\nRAW EXPECTED:{y_test}\n")

    result_df = pd.DataFrame(columns=['expected', 'prediction', 'compare', 'binary'])
    suc = 0
    for i in range(len(predictions)):
        prd = predictions.flat[i]
        exp = y_test.flat[i]
        cmp = compare_result_in_allowed_range(prd, exp)
        print(f'prediction: {prd}, expected: {exp}, comparison: {cmp}')
        if cmp:
            suc += 1
        result_df.loc[len(result_df.index)] = [exp, prd, cmp, int(cmp)]

    print(f"num of success: {suc}")
    print(f"num of fails: {len(predictions) - suc}")

    export_result(result_df, "single", TEST_FILE_ID)
