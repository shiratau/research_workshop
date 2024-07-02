from datetime import datetime
from lib import *

DEVIATION_PERCENTAGE = 20
UP_LIMIT = (100 + DEVIATION_PERCENTAGE) / 100
DOWN_LIMIT = (100 - DEVIATION_PERCENTAGE) / 100
NUMBER_OF_SD = 3


def compare_result_in_allowed_range(prediction, y_test_value) -> bool:
    return y_test_value * DOWN_LIMIT <= prediction <= y_test_value * UP_LIMIT


def parse_result_for_multi_sd_dataset(predictions, y_test):
    df = pd.DataFrame(columns=['expected', 'prediction', 'compare', 'binary'])
    suc = 0
    for i in range(len(predictions)):
        sd_prd_set = predictions[i]
        sd_exp_set = y_test[i]
        for j in range(len(sd_prd_set)):
            prd = sd_prd_set[j]
            exp = sd_exp_set[j]
            cmp = compare_result_in_allowed_range(prd, exp)
            print(f'set: {i}, prediction: {prd}, expected: {exp}, comparison: {cmp}')
            if cmp:
                suc += 1
            df.loc[len(df.index)] = [exp, prd, cmp, int(cmp)]

    print(f"num of success: {suc}")
    print(f"num of fails: {len(predictions)*NUMBER_OF_SD - suc}")
    return df


def set_and_get_wd():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)
    return this_dir


def export_result(df, folder, file_id):
    this_dir = set_and_get_wd()
    path = os.path.join(this_dir, 'results', folder, f'{file_id}_{int(datetime.now().timestamp())}.csv')
    df.to_csv(path)


