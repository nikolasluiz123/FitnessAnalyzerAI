import pandas as pd
from pandas import DataFrame


def get_dataframe_test_data() -> DataFrame:
    data_frame = pd.read_csv(r'C:\Users\nikol\git\IA\FitnessAnalyzerAI\data\workout_evolution\strong.csv')
    data_frame = remove_unused_columns(data_frame)
    data_frame = rename_columns(data_frame)

    convert_date_to_datetime(data_frame)

    return data_frame


def rename_columns(dataframe):
    dataframe = dataframe.rename(
        columns={
            'Date': 'data',
            'Workout Name': 'treino',
            'Exercise Name': 'exercicio',
            'Set Order': 'serie',
            'Weight': 'peso',
            'Reps': 'repeticoes'
        }
    )
    return dataframe


def remove_unused_columns(dataframe: DataFrame):
    return dataframe.drop(columns=['Distance', 'Seconds', 'Notes', 'Workout Notes', 'RPE', 'Duration'])


def convert_date_to_datetime(dataframe):
    dataframe['data'] = pd.to_datetime(dataframe['data'])
