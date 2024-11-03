import pandas as pd
from pandas import DataFrame


def get_dataframe_training_data() -> DataFrame:
    data_frame = pd.read_csv(r'C:\Users\nikol\git\IA\FitnessAnalyzerAI\data\workout_evolution\weightlifting_721_workouts.csv')
    data_frame = remove_unused_columns(data_frame)
    data_frame = rename_columns(data_frame)

    convert_weight_to_kg(data_frame)
    convert_date_to_datetime(data_frame)
    filter_dataframe_with_important_infos(data_frame)
    translate_exercises(data_frame)

    return data_frame


def rename_columns(dataframe):
    dataframe = dataframe.rename(
        columns={
            'Date': 'data',
            'Exercise Name': 'exercicio',
            'Set Order': 'serie',
            'Weight': 'peso',
            'Reps': 'repeticoes'
        }
    )
    return dataframe


def remove_unused_columns(dataframe: DataFrame):
    return dataframe.drop(columns=['Distance', 'Seconds', 'Notes', 'Workout Notes', 'Workout Name'])


def convert_weight_to_kg(dataframe: DataFrame):
    dataframe['peso'] = dataframe['peso'] * 0.453592


def convert_date_to_datetime(dataframe):
    dataframe['data'] = pd.to_datetime(dataframe['data'])
    # dataframe['data'] = dataframe['data'].values.astype("float64")


def filter_dataframe_with_important_infos(data_frame):
    data_frame.drop(index=data_frame[(data_frame['peso'] == 0) | (data_frame['repeticoes'] == 0)].index, inplace=True)

def translate_exercises(data_frame):
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

