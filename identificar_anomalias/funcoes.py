import pandas as pd
from sklearn.ensemble import IsolationForest


def detectar_anomalias_por_exercicio(df):
    anomalias = pd.DataFrame()
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)

    for exercicio in df['exercicio'].unique():
        df_exercicio = df[df['exercicio'] == exercicio].copy()
        features = df_exercicio[['repeticoes', 'peso', 'serie']]
        df_exercicio['anomalia'] = iso_forest.fit_predict(features)

        anomalias = pd.concat([anomalias, df_exercicio], ignore_index=True)

    anomalias = anomalias.sort_values(by=['exercicio'])

    return anomalias