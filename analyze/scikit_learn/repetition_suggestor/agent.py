import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnHalvingRandomCVHyperParamsSearcher, \
    ScikitLearnRandomCVHyperParamsSearcher
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
        self._data_pre_processor = ScikitLearnRepetitionSuggestorDataPreProcessor()

    def _initialize_multi_process_manager(self):
        params_searcher_decision_tree = ScikitLearnHalvingRandomCVHyperParamsSearcher(
            number_candidates=1500,
            min_resources=2000,
            max_resources=5000,
            resource='n_samples',
            log_level=1
        )

        params_searcher_random_forest = ScikitLearnHalvingRandomCVHyperParamsSearcher(
            number_candidates=250,
            min_resources=10,
            max_resources=100,
            resource='n_estimators',
            log_level=1
        )

        params_searcher_kneighbors = ScikitLearnHalvingRandomCVHyperParamsSearcher(
            number_candidates=1500,
            min_resources=2000,
            max_resources=5000,
            resource='n_samples',
            log_level=1
        )

        cross_validator = ScikitLearnCrossValidator(
            log_level=1
        )

        pipelines = [
            ScikitLearnPipeline(
                estimator=RandomForestRegressor(),
                params={
                    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'max_depth': randint(2, 30),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(2, 20),
                    'max_features': ['sqrt', 'log2', None]
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=None,
                params_searcher=params_searcher_random_forest,
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
                    'n_neighbors': randint(1, 20),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'brute'],
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=None,
                params_searcher=params_searcher_kneighbors,
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
                    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'max_depth': randint(2, 30),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(2, 20),
                    'max_features': ['sqrt', 'log2', None],
                    'splitter': ['best', 'random'],
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=None,
                params_searcher=params_searcher_decision_tree,
                history_manager=ScikitLearnCrossValidationHistoryManager(
                    output_directory='executions_history',
                    models_directory='models_decision_tree',
                    best_params_file_name='decision_tree_best_params',
                    cv_results_file_name='decision_tree_cv_results'
                ),
                validator=cross_validator
            )
        ]

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
            history_index=self.history_index
        )

    def _execute_additional_validation(self):
        if self._force_execute_additional_validation:
            validation_data = self._data_pre_processor.get_data_additional_validation()

            for pipe in self._process_manager.pipelines:
                if self.history_index is not None:
                    history_length = self.history_index + 1
                else:
                    history_length = pipe.history_manager.get_history_len()

                model = pipe.history_manager.get_saved_model(history_length)

                additional_validator = ScikitLearnRegressorAdditionalValidator(
                    estimator=model,
                    prefix_file_names=f'{type(model).__name__}_{history_length}',
                    validation_results_directory='additional_validations',
                    data=validation_data,
                    show_graphics=False
                )

                additional_validator.validate()

    def _execute_prediction(self, data_dictionary: dict):
        data_frame = pd.DataFrame.from_dict(data_dictionary, orient='columns')
        x = self._data_pre_processor.get_data_to_prediction(data_frame)

        history_best = ScikitLearnCrossValidationHistoryManager(
            output_directory='executions_history',
            models_directory='models_random_forest',
            best_params_file_name='random_forest_best_params',
            cv_results_file_name='random_forest_cv_results'
        )

        model = history_best.get_saved_model(version=5)

        return model.predict(x).round().astype(int)
