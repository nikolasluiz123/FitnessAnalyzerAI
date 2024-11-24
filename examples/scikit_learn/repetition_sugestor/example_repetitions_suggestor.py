import pandas as pd
from tabulate import tabulate

from analyze.scikit_learn.repetition_suggestor.agent import ScikitLearnRepetitionSuggestorAgent

suggestor = ScikitLearnRepetitionSuggestorAgent(
    force_execute_best_model_search=False,
    data_path='../../../data/workout_evolution/weightlifting_721_workouts.csv'
)

dict_teste = {
    'exercicio': ['Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra'],
    'peso': [40, 80, 90, 110, 120, 125],
    'serie': [1, 2, 3, 4, 5, 6],
}

predictions = suggestor.execute(dict_teste)

df = pd.DataFrame.from_dict(dict_teste, orient='columns')
df['repeticoes'] = predictions

print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))