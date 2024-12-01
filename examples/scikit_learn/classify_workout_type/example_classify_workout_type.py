import pandas as pd
from tabulate import tabulate

from analyze.scikit_learn.classify_workout_type.agent import ScikitLearnClassifyWorkoutTypeAgent

suggestor = ScikitLearnClassifyWorkoutTypeAgent(force_execute_best_model_search=False)

data = {
    'idade': [22, 35, 29, 40, 19, 27, 33, 45, 30, 26, 50, 38, 24, 31, 36, 28, 21, 44, 32, 23],
    'genero': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
               'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'peso': [70, 60, 75, 65, 80, 55, 68, 62, 78, 58, 85, 64, 72, 59, 73, 63, 77, 66, 71, 56],
    'altura': [1.75, 1.68, 1.80, 1.65, 1.82, 1.60, 1.78, 1.67, 1.83, 1.63, 1.85, 1.66, 1.76, 1.64, 1.79, 1.62, 1.81, 1.69, 1.74, 1.61],
    'duracao_sessao': [60, 45, 90, 50, 75, 40, 80, 55, 85, 30, 70, 35, 95, 48, 65, 52, 100, 38, 60, 42],
    'frequencia_treino': [5, 4, 6, 3, 5, 2, 6, 4, 5, 3, 5, 2, 6, 3, 5, 4, 6, 3, 5, 2],
    'nivel_experiencia': [2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2, 1],
    'percentual_gordura': [15, 20, 12, 22, 14, 21, 16, 23, 13, 19, 17, 18, 15, 24, 16, 20, 14, 22, 15, 18],
}

predictions = suggestor.execute(data)

df = pd.DataFrame.from_dict(data, orient='columns')
df['tipo_de_treino'] = predictions

print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))