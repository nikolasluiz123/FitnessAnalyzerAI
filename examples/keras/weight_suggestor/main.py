from keras.src.utils import plot_model
import keras
from wrappers.keras.history_manager.regressor_history_manager import KerasRegressorHistoryManager

history = KerasRegressorHistoryManager(output_directory='history_model_V3',
                                       models_directory='models',
                                       best_params_file_name='best_executions')

model = history.get_saved_model(1)

plot_model(
    model,
    to_file="modelo.png",  # Salva o diagrama em um arquivo
    show_shapes=True,      # Mostra as dimensões dos tensores
    show_layer_names=True, # Mostra os nomes das camadas
    expand_nested=True,    # Expande camadas aninhadas, se houver
    show_dtype=True,
    show_layer_activations=True,
    dpi=96,                 # Define a resolução da imagem
)

model.summary()