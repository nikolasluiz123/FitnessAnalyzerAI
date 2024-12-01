from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnRandomCVHyperParamsSearcher
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.additional_validator import ScikitLearnClassifierAdditionalValidator
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

from analyze.common.common_agent import CommonAgent
from analyze.scikit_learn.classify_workout_frequency.pre_processor import \
    ScikitLearnClassifyWorkoutFrequencyDataPreProcessor


class ScikitLearnClassifyWorkoutFrequencyAgent(CommonAgent):

    def _initialize_data_pre_processor(self):
        self._data_pre_processor = ScikitLearnClassifyWorkoutFrequencyDataPreProcessor()

    def _initialize_multi_process_manager(self):
        params_searcher = ScikitLearnRandomCVHyperParamsSearcher(number_iterations=500, log_level=1)
        cross_validator = ScikitLearnCrossValidator(log_level=1)

        pipelines = [
            ScikitLearnPipeline(
                estimator=DecisionTreeClassifier(),
                params={
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_depth': randint(2, 30),
                    'min_samples_split': randint(2, 30),
                    'min_samples_leaf': randint(1, 30),
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
            ),
            ScikitLearnPipeline(
                estimator=RandomForestClassifier(),
                params={
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'n_estimators': randint(10, 100),
                    'max_depth': randint(2, 30),
                    'min_samples_split': randint(2, 30),
                    'min_samples_leaf': randint(1, 30),
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
            scoring='accuracy',
            history_index=history_index,
            stratified=True
        )

    def _execute_additional_validation(self):
        if self._force_execute_best_model_search:
            validation_data = self._data_pre_processor.get_data_additional_validation()

            for pipe in self._process_manager.pipelines:
                model = pipe.history_manager.get_saved_model(pipe.history_manager.get_history_len())

                additional_validator = ScikitLearnClassifierAdditionalValidator(
                    estimator=model,
                    prefix_file_names=type(model).__name__,
                    validation_results_directory='additional_validations',
                    data=validation_data,
                    show_graphics=False,
                )

                additional_validator.validate()

    def _execute_prediction(self, data_dictionary: dict):
        predict_data = self._data_pre_processor.get_data_to_prediction(data_dictionary)

        history_len = self._process_manager.history_manager.get_history_len()
        model = self._process_manager.history_manager.get_saved_model(history_len)

        return model.predict(predict_data)