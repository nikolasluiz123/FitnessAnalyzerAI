import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnHalvingRandomCVHyperParamsSearcher
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.additional_validator import ScikitLearnRegressorAdditionalValidator
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

from analyze.common.common_agent import CommonAgent
from analyze.scikit_learn.repetition_suggestor.pre_processor import ScikitLearnRepetitionSuggestorDataPreProcessor


class ScikitLearnRepetitionSuggestorAgent(CommonAgent):
    """
    Agente responsável por realizar a sugestão de repetições de exercícios utilizando o melhor modelo encontrado do
    scikit-learn.
    """

    def _initialize_data_pre_processor(self):
        self._data_pre_processor = ScikitLearnRepetitionSuggestorDataPreProcessor(self.data_path)

    def _initialize_multi_process_manager(self):
        params_searcher = ScikitLearnHalvingRandomCVHyperParamsSearcher(number_candidates='exhaust',
                                                                        min_resources=100,
                                                                        max_resources=2000,
                                                                        resource='n_samples',
                                                                        factor=3,
                                                                        log_level=1)
        cross_validator = ScikitLearnCrossValidator(log_level=1)

        pipelines = [
            ScikitLearnPipeline(
                estimator=RandomForestRegressor(),
                params={
                    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                                     105, 110, 115, 120,
                                     125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200,
                                     205, 210, 215, 220,
                                     225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300,
                                     305, 310, 315,
                                     320],
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': [5, 10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
                    'max_features': ['sqrt', 'log2']
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=None,
                params_searcher=params_searcher,
                history_manager=ScikitLearnCrossValidationHistoryManager(
                    output_directory='executions_history',
                    models_directory='models_random_forest',
                    best_params_file_name='random_forest_best_params',
                    cv_results_file_name='random_forest_cv_results'
                ),
                validator=cross_validator
            ),
            ScikitLearnPipeline(
                estimator=KNeighborsRegressor(),
                params={
                    'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=None,
                params_searcher=params_searcher,
                history_manager=ScikitLearnCrossValidationHistoryManager(
                    output_directory='executions_history',
                    models_directory='models_kneighbors',
                    best_params_file_name='kneighbors_best_params',
                    cv_results_file_name='kneighbors_cv_results'
                ),
                validator=cross_validator
            ),
            ScikitLearnPipeline(
                estimator=DecisionTreeRegressor(),
                params={
                    'criterion': ['squared_error', 'absolute_error'],
                    'max_depth': [5, 10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
                    'max_features': ['sqrt', 'log2', None]
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=None,
                params_searcher=params_searcher,
                history_manager=ScikitLearnCrossValidationHistoryManager(
                    output_directory='executions_history',
                    models_directory='models_decision_tree',
                    best_params_file_name='decision_tree_best_params',
                    cv_results_file_name='decision_tree_cv_results'
                ),
                validator=cross_validator
            )
        ]

        if self._force_execute_best_model_search:
            history_index = None
        else:
            history_index = -1

        self._process_manager = ScikitLearnMultiProcessManager(
            pipelines=pipelines,
            history_manager=ScikitLearnCrossValidationHistoryManager(
                output_directory='best_executions_history',
                models_directory='best_models',
                best_params_file_name='best_params',
                cv_results_file_name='best_cv_results'
            ),
            fold_splits=10,
            scoring='neg_mean_absolute_error',
            history_index=history_index
        )

    def _execute_additional_validation(self):
        validation_data = self._data_pre_processor.get_data_additional_validation()

        for pipe in self._process_manager.pipelines:
            model = pipe.history_manager.get_saved_model(pipe.history_manager.get_history_len())

            additional_validator = ScikitLearnRegressorAdditionalValidator(
                estimator=model,
                prefix_file_names=type(model).__name__,
                validation_results_directory='additional_validations',
                data=validation_data,
                show_graphics=False
            )

            additional_validator.validate()

    def _execute_prediction(self, data_dictionary: dict):
        data_frame = pd.DataFrame.from_dict(data_dictionary, orient='columns')
        x = self._data_pre_processor.get_data_to_prediction(data_frame)

        history_len = self._process_manager.history_manager.get_history_len()
        model = self._process_manager.history_manager.get_saved_model(history_len)

        return model.predict(x).round().astype(int)
