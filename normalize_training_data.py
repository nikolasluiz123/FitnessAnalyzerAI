import pandas as pd
from pandas import DataFrame


def get_dataframe_training_data() -> DataFrame:
    data_frame = pd.read_csv(r'C:\Users\nikol\git\IA\FitnessAnalyzerAI\data\workout_evolution\weightlifting_721_workouts.csv')
    data_frame = remove_unused_columns(data_frame)
    data_frame = rename_columns(data_frame)

    convert_weight_to_kg(data_frame)
    convert_date_to_datetime(data_frame)
    filter_dataframe_with_important_infos(data_frame)

    return data_frame


def rename_columns(dataframe):
    dataframe = dataframe.rename(
        columns={
            'Date': 'data',
            'Exercise Name': 'exercicio',
            'Set Order': 'serie',
            'Weight': 'peso',
            'Reps': 'repeticoes'
        }
    )
    return dataframe


def remove_unused_columns(dataframe: DataFrame):
    return dataframe.drop(columns=['Distance', 'Seconds', 'Notes', 'Workout Notes', 'Workout Name'])


def convert_weight_to_kg(dataframe: DataFrame):
    dataframe['peso'] = dataframe['peso'] * 0.453592


def convert_date_to_datetime(dataframe):
    dataframe['data'] = pd.to_datetime(dataframe['data'])
    # dataframe['data'] = dataframe['data'].values.astype("float64")


def filter_dataframe_with_important_infos(data_frame):
    data_frame.drop(index=data_frame[(data_frame['peso'] == 0) | (data_frame['repeticoes'] == 0)].index, inplace=True)
