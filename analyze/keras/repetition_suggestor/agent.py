import pandas as pd
from keras.src.callbacks import EarlyStopping
from wrappers.keras.history_manager.regressor_history_manager import KerasRegressorHistoryManager
from wrappers.keras.hyper_params_search.hyper_band_searcher import KerasHyperBandSearcher
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.process_manager.regressor_multi_process_manager import KerasRegressorMultProcessManager
from wrappers.keras.validator.additional_validator import KerasAdditionalRegressorValidator
from wrappers.keras.validator.basic_regressor_validator import KerasBasicRegressorValidator

from analyze.common.common_agent import CommonAgent
from analyze.keras.repetition_suggestor.neural_networks import RepetitionSuggestorLSTMV1, RepetitionSuggestorLSTMV3, \
    RepetitionSuggestorLSTMV2
from analyze.keras.repetition_suggestor.pre_processor import KerasRepetitionsSuggestorDataPreProcessor


class KerasRepetitionSuggestorAgent(CommonAgent):
    """
    Agente responsável por realizar a sugestão de repetições de exercícios utilizando o melhor modelo de rede neural
    enconstrado.
    """

    def __init__(self, force_execute_best_model_search: bool, data_path: str):
        super().__init__(force_execute_best_model_search, data_path)

    def _initialize_data_pre_processor(self):
        self._data_pre_processor = KerasRepetitionsSuggestorDataPreProcessor(self.data_path)

    def _initialize_multi_process_manager(self):
        early_stopping_validation = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

        validator = KerasBasicRegressorValidator(
            epochs=200,
            batch_size=KerasRepetitionsSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[early_stopping_validation]
        )

        params_searcher_1 = KerasHyperBandSearcher(
            objective='val_loss',
            directory='search_params_1',
            project_name='model_example_1',
            epochs=15,
            batch_size=KerasRepetitionsSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[],
            max_epochs=20,
            factor=3,
            hyper_band_iterations=1
        )

        params_searcher_2 = KerasHyperBandSearcher(
            objective='val_loss',
            directory='search_params_2',
            project_name='model_example_2',
            epochs=15,
            batch_size=KerasRepetitionsSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[],
            max_epochs=20,
            factor=3,
            hyper_band_iterations=1
        )

        params_searcher_3 = KerasHyperBandSearcher(
            objective='val_loss',
            directory='search_params_3',
            project_name='model_example_3',
            epochs=15,
            batch_size=KerasRepetitionsSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[],
            max_epochs=20,
            factor=3,
            hyper_band_iterations=1
        )

        history_manager_model_example_1 = KerasRegressorHistoryManager(output_directory='history_model_example_1',
                                                                       models_directory='models',
                                                                       best_params_file_name='best_executions')

        history_manager_model_example_2 = KerasRegressorHistoryManager(output_directory='history_model_example_2',
                                                                       models_directory='models',
                                                                       best_params_file_name='best_executions')

        history_manager_model_example_3 = KerasRegressorHistoryManager(output_directory='history_model_example_3',
                                                                       models_directory='models',
                                                                       best_params_file_name='best_executions')

        pipelines = [
            KerasPipeline(
                model=RepetitionSuggestorLSTMV1(),
                data_pre_processor=self._data_pre_processor,
                params_searcher=params_searcher_1,
                validator=validator,
                history_manager=history_manager_model_example_1
            ),
            KerasPipeline(
                model=RepetitionSuggestorLSTMV2(),
                data_pre_processor=self._data_pre_processor,
                params_searcher=params_searcher_2,
                validator=validator,
                history_manager=history_manager_model_example_2
            ),
            KerasPipeline(
                model=RepetitionSuggestorLSTMV3(),
                data_pre_processor=self._data_pre_processor,
                validator=validator,
                params_searcher=params_searcher_3,
                history_manager=history_manager_model_example_3
            )
        ]

        history_manager_best_model = KerasRegressorHistoryManager(output_directory='best_executions',
                                                                  models_directory='best_models',
                                                                  best_params_file_name='best_executions')

        if self._force_execute_best_model_search:
            history_index = None
        else:
            history_index = -1

        self._process_manager = KerasRegressorMultProcessManager(
            pipelines=pipelines,
            seed=KerasRepetitionsSuggestorDataPreProcessor.SEED,
            history_manager=history_manager_best_model,
            history_index=history_index,
            save_history=self._force_execute_best_model_search,
            delete_trials_after_execution=True
        )

    def _execute_additional_validation(self):
        validation_data = self._data_pre_processor.get_data_additional_validation()
        validation_data = self._data_pre_processor.get_data_as_numpy(validation_data)

        for pipe in self._process_manager.pipelines:
            model = pipe.history_manager.get_saved_model(pipe.history_manager.get_history_len())

            additional_validator = KerasAdditionalRegressorValidator(model_instance=model,
                                                                     data=validation_data,
                                                                     prefix_file_names=type(model).__name__,
                                                                     validation_results_directory='additional_validations',
                                                                     show_graphics=False,
                                                                     scaler=self._data_pre_processor.scaler_y)
            additional_validator.validate()

    def _execute_prediction(self, dataset):
        history_len = self._process_manager.history_manager.get_history_len()
        model = self._process_manager.history_manager.get_saved_model(history_len)

        predictions = model.predict(self._data_pre_processor.get_data_as_numpy(dataset))

        return self._data_pre_processor.scaler_y.inverse_transform(predictions).round().astype(int)