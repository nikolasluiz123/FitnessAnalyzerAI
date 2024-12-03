import pandas as pd
from tabulate import tabulate

from analyze.scikit_learn.weight_suggestor.agent import ScikitLearnWeightSuggestorAgent

suggestor = ScikitLearnWeightSuggestorAgent(
    history_index=None,
    force_execute_additional_validation=True
)

dict_teste = {
    'exercicio': ['Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra', 'Supino com Barra'],
    'repeticoes': [15, 12, 10, 8, 6, 4],
    'serie': [1, 2, 3, 4, 5, 6],
}

predictions = suggestor.execute(dict_teste)

df = pd.DataFrame.from_dict(dict_teste, orient='columns')
df['peso'] = predictions

print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
