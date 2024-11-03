import time
from datetime import date

from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from identificar_anomalias.funcoes import detectar_anomalias_por_exercicio
from normalize_training_data import get_dataframe_training_data

df = get_dataframe_training_data()

label_encoder = LabelEncoder()
df['exercicio'] = label_encoder.fit_transform(df['exercicio'])

df = df.groupby(['data', 'exercicio']).agg(
    serie=('serie', 'count'),
    peso=('peso', 'mean'),
    repeticoes=('repeticoes', 'sum')
).reset_index()

df_anomalias = detectar_anomalias_por_exercicio(df)
# print('Anomalias Detectadas (-1 Fora do Padrão, 1 No Padrão):')
# print(tabulate(df_anomalias, headers='keys', tablefmt='fancy_grid', showindex=False))

print('Anomalias Detectadas Exercício 0:')
df_exercicio_0 = df_anomalias.query('exercicio == 0')
df_exercicio_0 = df_exercicio_0.sort_values(by=['serie', 'peso', 'repeticoes'], ascending=False)

print(tabulate(df_exercicio_0, headers='keys', tablefmt='fancy_grid', showindex=False))