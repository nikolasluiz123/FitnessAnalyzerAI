import pandas as pd
from pandas import DataFrame


def get_dataframe_from_cv_results(cv_results: dict, tested_params: dict) -> DataFrame:
    columns = ['rank_test_score', 'mean_test_score', 'std_test_score']

    params = [p for p in tested_params.keys()]

    for p in params:
        columns.append(f'param_{p}')

    df = pd.DataFrame.from_dict(cv_results, orient='columns')
    df.drop(columns=['params'], inplace=True)
    df = df.sort_values(by=['rank_test_score'])

    return df