import pandas as pd
from tabulate import tabulate

from analyze.keras.weight_suggestor.agent import KerasWeightSuggestorAgent
from analyze.keras.weight_suggestor.pre_processor import KerasWeightSuggestorDataPreProcessor

suggestor = KerasWeightSuggestorAgent(
    history_index=0,
    force_execute_additional_validation=True
)

test_data_pre_processor = KerasWeightSuggestorDataPreProcessor()

data = test_data_pre_processor.get_data_additional_validation()
predictions = suggestor.execute(data)

x_test, y_test = test_data_pre_processor.get_test_data_to_create_dataframe()

df = pd.DataFrame(x_test, columns=['exercicio', 'serie', 'repeticoes', 'dia_da_semana', 'dia_desde_inicio'])
df['peso'] = y_test
df['peso_sugerido'] = predictions

print(tabulate(df.head(20), headers='keys', tablefmt='fancy_grid', showindex=False))