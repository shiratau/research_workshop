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
    samples_input = [np.array(ast.literal_eval(sample), dtype=float) for sample in raw_df["sample"]]
    sd_output = [_ for _ in raw_df["sd"]]

    df = pd.DataFrame(
        {
            "input": pd.Series(samples_input),
            "output": sd_output
        }
    )

    predictions, y_test = predict_multi_sd.run(df)

    suc = 0
    for i in range(len(predictions)):
        prd = predictions.flat[i]
        exp = y_test.flat[i]
        cmp = _compare_result_in_allowed_range(prd, exp)
        print(f'prediction: {prd}, expected: {exp}, comparison: {cmp}')
        if cmp:
            suc += 1
    print(f"num of success: {suc}")
    print(f"num of fails: {len(predictions) - suc}")


def _compare_result_in_allowed_range(prediction, y_test_value) -> bool:
    return y_test_value * DOWN_LIMIT <= prediction <= y_test_value * UP_LIMIT
