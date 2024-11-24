import pandas as pd
from tabulate import tabulate

from analyze.keras.repetition_suggestor.agent import KerasRepetitionSuggestorAgent
from analyze.keras.repetition_suggestor.pre_processor import KerasRepetitionsSuggestorDataPreProcessor

suggestor = KerasRepetitionSuggestorAgent(
    force_execute_best_model_search=False,
    data_path='../../../data/workout_evolution/weightlifting_721_workouts.csv'
)

test_data_pre_processor = KerasRepetitionsSuggestorDataPreProcessor(
    data_path='../../../data/workout_evolution/weightlifting_721_workouts.csv'
)

data = test_data_pre_processor.get_data_additional_validation()
predictions = suggestor.execute(data)

x_test, y_test = test_data_pre_processor.get_test_data_to_create_dataframe()

df = pd.DataFrame(x_test, columns=['exercicio', 'serie', 'peso', 'dia_da_semana', 'dia_desde_inicio'])
df['repeticoes'] = y_test
df['repeticoes_sugerida'] = predictions

print(tabulate(df.head(20), headers='keys', tablefmt='fancy_grid', showindex=False))