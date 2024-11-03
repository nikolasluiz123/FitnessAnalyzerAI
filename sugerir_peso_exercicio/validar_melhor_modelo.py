import pandas as pd
from scikit_learn.history_manager.cross_validator import CrossValidatorHistoryManager
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from normalize_training_data import get_dataframe_training_data

df = get_dataframe_training_data()

label_encoder = LabelEncoder()
label_encoder.fit(df['exercicio'])

df_supino_barra = df.query('exercicio == "Supino com Barra"')
print(tabulate(df_supino_barra, headers='keys', tablefmt='fancy_grid', showindex=False))

dict_teste = {
    'exercicio': ['Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra'],
    'serie': [1, 2, 3, 4, 5, 6],
    'repeticoes': [15, 12, 10, 8, 6, 4]
}

df_teste = pd.DataFrame.from_dict(dict_teste, orient='columns')
df_teste['exercicio'] = label_encoder.transform(df_teste['exercicio'])

print(tabulate(df_teste, headers='keys', tablefmt='fancy_grid', showindex=False))

best_params_history_manager = CrossValidatorHistoryManager(output_directory='best_results',
                                                           models_directory='best_models',
                                                           params_file_name='best_params',
                                                           cv_results_file_name='cv_results')

model = best_params_history_manager.get_saved_model(2)
weights = model.predict(df_teste)

df_teste['peso'] = weights

print(tabulate(df_teste, headers='keys', tablefmt='fancy_grid', showindex=False))