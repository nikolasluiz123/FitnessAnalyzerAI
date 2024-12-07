import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from wrappers.common.data_pre_processor.common_data_pre_processor import CommonDataPreProcessor


class KerasRepetitionsSuggestorDataPreProcessor(CommonDataPreProcessor):
    SEQUENCE_LENGTH = 7
    """
    Indica a quantidade de registros que serão usados como entrada para a rede neural. Especificamente nesse contexto
    podemos considerar que os dados serão enviados em sequências de 7 dias.
    """

    FEATURES_NUMBER = 5
    """
    Indica quantas features serão utilizadas para sugerir o valor desejado.
    """

    BATCH_SIZE = 256
    """
    Tamanho do batch utilizado nos processos de treinamento das redes neurais.
    """

    SEED = 42
    """
    Valor da seed utilizado nos processos que possuem alguma lógica de randomização.
    """

    def __init__(self):
        super().__init__()
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _on_execute_train_process(self) -> tuple:
        x_test, x_train, y_test, y_train = self.get_train_test_data()

        train_dataset = self._create_tf_dataset(x_train, y_train)
        val_dataset = self._create_tf_dataset(x_test, y_test)

        return train_dataset, val_dataset

    def get_data_additional_validation(self):
        x_test, x_train, y_test, y_train = self.get_train_test_data()
        val_dataset = self._create_tf_dataset(x_test, y_test)

        return val_dataset

    def _create_tf_dataset(self, x_data, y_data):
        """
        Função utilizada para criar um dataset do Tensorflow para que os processos realizados no treinamento da rede
        funcionem como esperado.

        :param x_data: Valores utilizados para previsão
        :param y_data: Valores daquilo que queremos prever e que sabemos que está correto
        """
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = (
            dataset
            .shuffle(buffer_size=len(x_data), seed=self.SEED)
            .batch(self.BATCH_SIZE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return dataset

    def get_train_test_data(self):
        """
        Função para obter os dados subdivididos em treino e teste realizando todos os tratamentos julgados como
        necessários.
        """
        path = kagglehub.dataset_download("joep89/weightlifting")

        data_frame = pd.read_csv(f'{path}/weightlifting_721_workouts.csv')
        data_frame = self.__remove_unused_columns(data_frame)
        data_frame = self.__rename_columns(data_frame)

        self.__convert_weight_to_kg(data_frame)
        self.__convert_date_to_datetime(data_frame)

        data_frame = self.__create_data_informations(data_frame)
        self.__filter_dataframe_with_important_infos(data_frame)

        data_frame = self.__remove_invalid_exercises_and_translate(data_frame)
        self.__encoding_exercises(data_frame)
        self._drop_null_exercises(data_frame)

        x_scaled, y_scaled = self.__get_features_and_target(data_frame)
        x, y = self.create_sequences(x_scaled, y_scaled, self.SEQUENCE_LENGTH)

        split_index = int(0.8 * len(x))
        x_train, x_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return x_test, x_train, y_test, y_train

    def get_test_data_to_create_dataframe(self):
        path = kagglehub.dataset_download("joep89/weightlifting")

        data_frame = pd.read_csv(f'{path}/weightlifting_721_workouts.csv')
        data_frame = self.__remove_unused_columns(data_frame)
        data_frame = self.__rename_columns(data_frame)

        self.__convert_weight_to_kg(data_frame)
        self.__convert_date_to_datetime(data_frame)

        data_frame = self.__create_data_informations(data_frame)
        self.__filter_dataframe_with_important_infos(data_frame)

        data_frame = self.__remove_invalid_exercises_and_translate(data_frame)
        self.__encoding_exercises(data_frame)
        self._drop_null_exercises(data_frame)

        features = data_frame[self.get_features_list()].values
        target = data_frame[self.get_target()].values.reshape(-1, 1)

        split_index = int(0.8 * len(features)) + 1
        x_train, x_test = features[:split_index], features[split_index:]
        y_train, y_test = target[:split_index], target[split_index:]

        return x_test, y_test

    @staticmethod
    def __remove_unused_columns(data_frame: DataFrame):
        """
        Remove as colunas que não fazem diferença para a previsão.

        :param data_frame: DataFrame que será aplicado o processamento
        """
        return data_frame.drop(columns=['Distance', 'Seconds', 'Notes', 'Workout Notes', 'Workout Name'])

    @staticmethod
    def __rename_columns(data_frame: DataFrame):
        """
        Função que renomeia as colunas para facilitar o entendimento e acesso aos campos no código

        :param data_frame: DataFrame que será aplicado o processamento
        """

        data_frame = data_frame.rename(
            columns={
                'Date': 'data',
                'Exercise Name': 'exercicio',
                'Set Order': 'serie',
                'Weight': 'peso',
                'Reps': 'repeticoes'
            }
        )

        return data_frame

    @staticmethod
    def __convert_weight_to_kg(data_frame: DataFrame):
        """
        Função que realiza a conversão do peso para KG.

        :param data_frame: DataFrame que será aplicado o processamento
        """
        data_frame['peso'] = data_frame['peso'] * 0.453592

    @staticmethod
    def __convert_date_to_datetime(data_frame: DataFrame):
        """
        Função para realizar a conversão da data em um tipo que possa ser manipulado posteriormente no python.

        :param data_frame: DataFrame que será aplicado o processamento
        """

        data_frame['data'] = pd.to_datetime(data_frame['data'])

    @staticmethod
    def __filter_dataframe_with_important_infos(data_frame: DataFrame):
        """
        Função para eliminar alguns registros que foram feitos de exercícios que não possuiam peso ou repetições, seja
        por erro de registro da informação ou pelo fato do exercício não se enquandrar nesse padrão.

        :param data_frame: DataFrame que será aplicado o processamento
        """

        filter_condition = (data_frame['peso'] == 0) | (data_frame['repeticoes'] == 0)
        data_frame.drop(index=data_frame[filter_condition].index, inplace=True)

    @staticmethod
    def __remove_invalid_exercises_and_translate(data_frame: DataFrame):
        """
        Por serem poucos exercícios foi realizada a tradução deles e, aqueles que não são tão comuns, foram removidos
        para reduzir um pouco a variedade.

        :param data_frame: DataFrame que será aplicado o processamento
        """
        translation = {
            'Bench Press (Barbell)': 'Supino com Barra',
            'Bent Over Row (Dumbbell)': 'Remada Curvada com Halteres',
            'Bicep Curl (Barbell)': 'Rosca Bíceps com Barra',
            'Bicep Curl (Dumbbell)': 'Rosca Bíceps com Halteres',
            'Cable Fly': 'Crucifixo no Cabo',
            'Chin Up': 'Barra Fixa Supinada',
            'Curl Dumbbell': 'Rosca com Halteres',
            'Deadlift': 'Levantamento Terra',
            'Deadlift (Barbell)': 'Levantamento Terra com Barra',
            'Deadlift - Trap Bar': 'Levantamento Terra (Trap Bar)',
            'Face pull': 'Puxada Face Pull',
            'Front Raise (Dumbbell)': 'Elevação Frontal com Halteres',
            'Front Squat (Barbell)': 'Agachamento Frontal com Barra',
            'Glute extension': 'Extensão de Glúteo',
            'Good Morning': 'Good Morning',
            'Good Morning (Barbell)': 'Good Morning (Barra)',
            'Hack Squat': 'Agachamento no Hack',
            'Hammer Curl': 'Rosca Martelo',
            'Hammer Curl (Dumbbell)': 'Rosca Martelo com Haltere',
            'Hammer Decline Chest Press': 'Supino Declinado (Máquina)',
            'Hammer High Row - 1 Arm': 'Remada Alta (Máquina, 1 Braço)',
            'Hammer Row - Wide Grip': 'Remada (Pegada Larga, Máquina)',
            'Hammer Row Stand 1armed': 'Remada em Pé (1 Braço, Máquina)',
            'Hammer back row wide 45 angle': 'Remada 45º (Pegada Larga, Máquina)',
            'Hammer lat pulldown': 'Pulldown (Máquina)',
            'Hammer seated row': 'Remada Sentado (Máquina)',
            'Hammer seated row (CLOSE GRIP)': 'Remada Sentado (Pegada Fechada, Máquina)',
            'Hammer shoulder press': 'Desenvolvimento na Máquina',
            'Incline Bench Press': 'Supino Inclinado',
            'Incline Bench Press (Barbell)': 'Supino Inclinado com Barra',
            'Incline Press (Dumbbell)': 'Supino Inclinado com Halteres',
            'Landmine Press': 'Landmine Press',
            'Lat Pulldown': 'Pulldown',
            'Lat Pulldown (Cable)': 'Pulldown no Cabo',
            'Lat Pulldown Closegrip': 'Pulldown (Pegada Fechada)',
            'Lateral Raise': 'Elevação Lateral',
            'Lateral Raise (Dumbbell)': 'Elevação Lateral com Halteres',
            'Leg Extension (Machine)': 'Cadeira Extensora',
            'Leg Curl': 'Mesa Flexora',
            'Leg Outward Fly': 'Abdutora',
            'Leg Press': 'Leg Press',
            'Leg Press (hinge)': 'Leg Press (Dobradiça)',
            'Low Incline Dumbbell Bench': 'Supino Inclinado Baixo (Halter)',
            'Military Press (Standing)': 'Desenvolvimento Militar (Em Pé)',
            'Neutral Chin': 'Barra Neutra',
            'Overhead Press (Barbell)': 'Desenvolvimento (Barra)',
            'Overhead Press (Dumbbell)': 'Desenvolvimento (Halter)',
            'Pull Up': 'Barra Fixa Pronada',
            'Rack Pull - 1 Pin': 'Rack Pull (1 Pino)',
            'Rack Pull 2 Pin': 'Rack Pull (2 Pinos)',
            'Rear delt fly': 'Crucifixo Invertido',
            'Romanian Deadlift (Barbell)': 'Levantamento Terra Romeno (Barra)',
            'Rope Never Ending': 'Corda Infinita',
            'Rotator Cuff Work': 'Exercício para Manguito Rotador',
            'Seated Cable Row (close Grip)': 'Remada Sentada (Pegada Fechada, Cabo)',
            'Seated Military Press': 'Desenvolvimento Militar Sentado',
            'Seated Military Press (Dumbbell)': 'Desenvolvimento Militar Sentado (Halter)',
            'Seated Row': 'Remada Sentada',
            'Seated Shoulder Press (Barbell)': 'Desenvolvimento Sentado (Barra)',
            'Seated Shoulder Press (Dumbbell)': 'Desenvolvimento Sentado (Halter)',
            'Shoulder Press (Standing)': 'Desenvolvimento (Em Pé)',
            'Shrugs': 'Encolhimento',
            'Shrugs (dumbbell)': 'Encolhimento com Halteres',
            'Skullcrusher (Barbell)': 'Tríceps Testa com Barra',
            'Sling Shot Bench': 'Supino com Sling Shot',
            'Sling Shot Incline': 'Supino Inclinado com Sling Shot',
            'Squat': 'Agachamento',
            'Squat (Barbell)': 'Agachamento com Barra',
            'T-bar Row': 'Remada Cavalinho',
            'Tricep Extension': 'Extensão de Tríceps',
            'Tricep Pushdown': 'Pushdown de Tríceps',
            'Weighted dips': 'Mergulho com Peso',
            'Close Grip Bench': 'Supino Pegada Fechada',
            'Curl EZ Bar': 'Rosca EZ',
            'High Bar Squat': 'Agachamento High Bar',
            'Kettlebell Swings': 'Swing com Kettlebell',
            'Lying Skullcrusher': 'Tríceps Testa deitado',
            'Sumo Deadlift': 'Levantamento Terra Sumo'
        }

        data_frame['exercicio'] = data_frame['exercicio'].map(translation)

        invalid_exercises = [
            'Rosca com Halteres', 'Levantamento Terra', 'Levantamento Terra (Trap Bar)', 'Puxada Face Pull',
            'Good Morning', 'Good Morning (Barra)', 'Rosca Martelo', 'Supino Declinado (Máquina)',
            'Remada Alta (Máquina, 1 Braço)', 'Remada (Pegada Larga, Máquina)', 'Remada em Pé (1 Braço, Máquina)',
            'Remada 45º (Pegada Larga, Máquina)', 'Pulldown (Máquina)', 'Remada Sentado (Máquina)',
            'Remada Sentado (Pegada Fechada, Máquina)', 'Supino Inclinado', 'Landmine Press', 'Pulldown',
            'Pulldown (Pegada Fechada)', 'Elevação Lateral', 'Leg Press (Dobradiça)',
            'Supino Inclinado Baixo (Halter)', 'Desenvolvimento Militar (Em Pé)', 'Barra Neutra',
            'Desenvolvimento (Barra)', 'Desenvolvimento (Halter)', 'Rack Pull (1 Pino)', 'Rack Pull (2 Pinos)',
            'Levantamento Terra Romeno (Barra)', 'Corda Infinita', 'Exercício para Manguito Rotador',
            'Remada Sentada (Pegada Fechada, Cabo)', 'Desenvolvimento Militar Sentado',
            'Desenvolvimento Militar Sentado (Halter)', 'Remada Sentada', 'Desenvolvimento Sentado (Barra)',
            'Desenvolvimento Sentado (Halter)', 'Desenvolvimento (Em Pé)', 'Encolhimento',
            'Supino com Sling Shot', 'Supino Inclinado com Sling Shot', 'Agachamento',
            'Extensão de Tríceps', 'Pushdown de Tríceps', 'Mergulho com Peso', 'Supino Pegada Fechada',
            'Rosca EZ', 'Agachamento High Bar', 'Swing com Kettlebell', 'Tríceps Testa deitado'
        ]

        data_frame = data_frame[~data_frame['exercicio'].isin(invalid_exercises)]

        return data_frame

    @staticmethod
    def _drop_null_exercises(data_frame: DataFrame):
        """
        Realiza a remooção de linhas que o exercício não tenha sido preenchido por algum motivo.

        :param data_frame: DataFrame que será aplicado o processamento
        """
        data_frame.dropna(subset=['exercicio'], inplace=True)

    def __create_data_informations(self, data_frame: DataFrame):
        """
        Realiza a segregação das informações da data em diferentes campos relevantes para a previsão.

        :param data_frame: DataFrame que será aplicado o processamento
        """

        df = data_frame.sort_values('data')

        df['dia_da_semana'] = df['data'].dt.weekday
        df['dias_desde_inicio'] = (df['data'] - df['data'].min()).dt.days

        return df

    def __encoding_exercises(self, data_frame: DataFrame):
        """
        Função para realizar o encoding dos exercícios já que a rede não entende dados textuais

        :param data_frame: DataFrame que será aplicado o processamento
        """

        label_encoder = LabelEncoder()
        data_frame['exercicio'] = label_encoder.fit_transform(data_frame['exercicio'])

    def __get_features_and_target(self, data_frame: DataFrame):
        """
        Função usada para obter os dados das features e target escalados

        :param data_frame: DataFrame que será aplicado o processamento
        """
        features = data_frame[self.get_features_list()].values
        target = data_frame[self.get_target()].values.reshape(-1, 1)

        features_scaled = self.scaler_x.fit_transform(features)
        target_scaled = self.scaler_y.fit_transform(target)

        return features_scaled, target_scaled

    @staticmethod
    def create_sequences(data, target, sequence_length):
        """
        Função utilizada para criar as sequêncas com os registros respeitando a quantidade definida

        :param data: Dados das features
        :param target: Dados do target
        :param sequence_length: Quantidade da sequência
        """
        x, y = [], []

        for i in range(len(data) - sequence_length):
            x.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])

        return np.array(x), np.array(y)

    def get_data_as_numpy(self, val_dataset):
        x_data, y_data = zip(*[(x.numpy(), y.numpy()) for x, y in val_dataset])
        x_data = np.concatenate(x_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        return x_data, y_data

    def prepare_for_predict(self, dataset):
        self._on_execute_train_process()
        return self.get_data_as_numpy(dataset)

    def get_features_list(self) -> list[str]:
        return ['exercicio', 'serie', 'peso', 'dia_da_semana', 'dias_desde_inicio']

    def get_target(self) -> str:
        return 'repeticoes'
