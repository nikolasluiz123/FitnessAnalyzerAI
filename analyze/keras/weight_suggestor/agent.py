from keras.src.callbacks import EarlyStopping
from wrappers.keras.history_manager.regressor_history_manager import KerasRegressorHistoryManager
from wrappers.keras.hyper_params_search.hyper_band_searcher import KerasHyperBandSearcher
from wrappers.keras.process_manager.pipeline import KerasPipeline
from wrappers.keras.process_manager.regressor_multi_process_manager import KerasRegressorMultProcessManager
from wrappers.keras.validator.additional_validator import KerasAdditionalRegressorValidator
from wrappers.keras.validator.basic_regressor_validator import KerasBasicRegressorValidator

from analyze.common.common_agent import CommonAgent
from analyze.keras.weight_suggestor.neural_networks import WeightSuggestorLSTMV1, WeightSuggestorLSTMV2, \
    WeightSuggestorLSTMV3
from analyze.keras.weight_suggestor.pre_processor import KerasWeightSuggestorDataPreProcessor


class KerasWeightSuggestorAgent(CommonAgent):
    """
    Agente responsável por realizar a sugestão de carga de exercícios utilizando o melhor modelo de rede neural
    enconstrado.
    """

    def _initialize_data_pre_processor(self):
        self._data_pre_processor = KerasWeightSuggestorDataPreProcessor()

    def _initialize_multi_process_manager(self):
        early_stopping_validation = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

        validator = KerasBasicRegressorValidator(
            epochs=300,
            batch_size=KerasWeightSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[early_stopping_validation]
        )

        params_searcher_v1 = KerasHyperBandSearcher(
            objective='val_loss',
            directory='search_params_V1',
            project_name='model_V1',
            epochs=30,
            batch_size=KerasWeightSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[],
            max_epochs=30,
            factor=3,
            hyper_band_iterations=1
        )

        params_searcher_v2 = KerasHyperBandSearcher(
            objective='val_loss',
            directory='search_params_V2',
            project_name='model_V2',
            epochs=30,
            batch_size=KerasWeightSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[],
            max_epochs=30,
            factor=3,
            hyper_band_iterations=1
        )

        params_searcher_v3 = KerasHyperBandSearcher(
            objective='val_loss',
            directory='search_params_V3',
            project_name='model_V3',
            epochs=30,
            batch_size=KerasWeightSuggestorDataPreProcessor.BATCH_SIZE,
            log_level=1,
            callbacks=[],
            max_epochs=30,
            factor=3,
            hyper_band_iterations=1
        )

        history_manager_model_v1 = KerasRegressorHistoryManager(output_directory='history_model_V1',
                                                                models_directory='models',
                                                                best_params_file_name='best_executions')

        history_manager_model_v2 = KerasRegressorHistoryManager(output_directory='history_model_V2',
                                                                models_directory='models',
                                                                best_params_file_name='best_executions')

        history_manager_model_v3 = KerasRegressorHistoryManager(output_directory='history_model_V3',
                                                                models_directory='models',
                                                                best_params_file_name='best_executions')

        pipelines = [
            KerasPipeline(
                model=WeightSuggestorLSTMV1(),
                data_pre_processor=self._data_pre_processor,
                params_searcher=params_searcher_v1,
                validator=validator,
                history_manager=history_manager_model_v1
            ),
            KerasPipeline(
                model=WeightSuggestorLSTMV2(),
                data_pre_processor=self._data_pre_processor,
                params_searcher=params_searcher_v2,
                validator=validator,
                history_manager=history_manager_model_v2
            ),
            KerasPipeline(
                model=WeightSuggestorLSTMV3(),
                data_pre_processor=self._data_pre_processor,
                params_searcher=params_searcher_v3,
                validator=validator,
                history_manager=history_manager_model_v3
            )
        ]

        history_manager_best_model = KerasRegressorHistoryManager(output_directory='best_executions',
                                                                  models_directory='best_models',
                                                                  best_params_file_name='best_executions')

        self._process_manager = KerasRegressorMultProcessManager(
            pipelines=pipelines,
            seed=KerasWeightSuggestorDataPreProcessor.SEED,
            history_manager=history_manager_best_model,
            history_index=self.history_index,
            delete_trials_after_execution=True,
            save_history=self.history_index is None
        )

    def _execute_additional_validation(self):
        if self._force_execute_additional_validation:
            validation_data = self._data_pre_processor.get_data_additional_validation()
            validation_data = self._data_pre_processor.get_data_as_numpy(validation_data)

            for pipe in self._process_manager.pipelines:
                if self.history_index is not None:
                    history_length = self.history_index + 1
                else:
                    history_length = pipe.history_manager.get_history_len()

                model = pipe.history_manager.get_saved_model(history_length)

                additional_validator = KerasAdditionalRegressorValidator(model_instance=model,
                                                                         data=validation_data,
                                                                         prefix_file_names=f'{type(pipe.model).__name__}_{history_length}',
                                                                         validation_results_directory='additional_validations',
                                                                         show_graphics=False,
                                                                         scaler=self._data_pre_processor.scaler_y)
                additional_validator.validate()

    def _execute_prediction(self, dataset):
        history_best = KerasRegressorHistoryManager(output_directory='history_model_V3',
                                                    models_directory='models',
                                                    best_params_file_name='best_executions')

        model = history_best.get_saved_model(version=1)
        predictions = model.predict(self._data_pre_processor.prepare_for_predict(dataset))

        return self._data_pre_processor.scaler_y.inverse_transform(predictions).round().astype(int)
