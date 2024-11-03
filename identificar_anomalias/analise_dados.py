from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from normalize_training_data import get_dataframe_training_data

df = get_dataframe_training_data()

label_encoder = LabelEncoder()
df['exercicio'] = label_encoder.fit_transform(df['exercicio'])
df = df.sort_values(by=['exercicio'])

print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))