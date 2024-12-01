import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor

class ScikitLearnClassifyWorkoutFrequencyDataPreProcessor(CommonDataPreProcessor):

    SEED = 42

    def __init__(self):
        super().__init__()

        self.tipo_treino_encoder = LabelEncoder()
        self.genero_encoder = LabelEncoder()

    def _on_execute_train_process(self) -> tuple:
        x, y = self.get_features_and_target_to_train()

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=self.SEED,
                                                            stratify=y)

        return x_train, y_train

    def get_data_additional_validation(self):
        x, y = self.get_features_and_target_to_train()

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=self.SEED,
                                                            stratify=y)

        return x_test, y_test

    def get_features_and_target_to_train(self):
        path = kagglehub.dataset_download(handle="valakhorasani/gym-members-exercise-dataset")
        df = pd.read_csv(f'{path}/gym_members_exercise_tracking.csv', sep=',')

        df.columns = ['idade', 'genero', 'peso', 'altura', 'batimento_maximo', 'batimento_medio', 'batimento_descanso',
                      'duracao_sessao', 'calorias_queimadas', 'tipo_treino', 'percentual_gordura', 'consumo_agua',
                      'frequencia_treino', 'nivel_experiencia', 'batimentos_minuto']

        df['genero'] = self.genero_encoder.fit_transform(df['genero'])
        df['tipo_treino'] = self.tipo_treino_encoder.fit_transform(df['tipo_treino'])

        x = df[['idade', 'genero', 'peso', 'altura', 'duracao_sessao', 'tipo_treino', 'nivel_experiencia', 'percentual_gordura']]
        y = df['frequencia_treino']

        return x, y

    def get_data_to_prediction(self, dictionary: dict):
        self._on_execute_train_process()

        df = pd.DataFrame.from_dict(dictionary, orient='columns')

        df['genero'] = self.genero_encoder.transform(df['genero'])
        df['tipo_treino'] = self.tipo_treino_encoder.transform(df['tipo_treino'])

        return df