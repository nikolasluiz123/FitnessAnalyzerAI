from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from wrappers.scikit_learn.features_search.rfe_searcher import ScikitLearnRecursiveFeatureCVSearcher
from wrappers.scikit_learn.features_search.select_k_best_searcher import ScikitLearnSelectKBestSearcher
from wrappers.scikit_learn.history_manager.cross_validation_history_manager import \
    ScikitLearnCrossValidationHistoryManager
from wrappers.scikit_learn.hyper_params_search.random_searcher import ScikitLearnRandomCVHyperParamsSearcher, \
    ScikitLearnHalvingRandomCVHyperParamsSearcher
from wrappers.scikit_learn.process_manager.multi_process_manager import ScikitLearnMultiProcessManager
from wrappers.scikit_learn.process_manager.pipeline import ScikitLearnPipeline
from wrappers.scikit_learn.validator.additional_validator import ScikitLearnClassifierAdditionalValidator
from wrappers.scikit_learn.validator.cross_validator import ScikitLearnCrossValidator

from analyze.common.common_agent import CommonAgent
from analyze.scikit_learn.classify_workout_type.pre_processor import ScikitLearnClassifyWorkoutTypeDataPreProcessor


class ScikitLearnClassifyWorkoutTypeAgent(CommonAgent):

    def _initialize_data_pre_processor(self):
        self._data_pre_processor = ScikitLearnClassifyWorkoutTypeDataPreProcessor()

    def _initialize_multi_process_manager(self):
        recursive_feature_cv_searcher = ScikitLearnRecursiveFeatureCVSearcher(log_level=1)
        select_k_best_searcher = ScikitLearnSelectKBestSearcher(features_number=5, score_func=f_classif, log_level=1)
        params_searcher_decision_tree = ScikitLearnHalvingRandomCVHyperParamsSearcher(
            number_candidates=1000,
            min_resources=300,
            max_resources=700,
            resource='n_samples',
            log_level=1
        )

        params_searcher_random_forest = ScikitLearnHalvingRandomCVHyperParamsSearcher(
            number_candidates=1000,
            min_resources=100,
            max_resources=200,
            resource='n_estimators',
            log_level=1
        )

        params_searcher_kneighbors = ScikitLearnHalvingRandomCVHyperParamsSearcher(
            number_candidates=1000,
            min_resources=100,
            max_resources=700,
            resource='n_samples',
            log_level=1
        )

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
                feature_searcher=recursive_feature_cv_searcher,
                params_searcher=params_searcher_decision_tree,
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
                    'max_depth': randint(2, 30),
                    'min_samples_split': randint(2, 30),
                    'min_samples_leaf': randint(1, 30),
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=recursive_feature_cv_searcher,
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
                estimator=KNeighborsClassifier(),
                params={
                    'n_neighbors': randint(1, 10),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'brute'],
                },
                data_pre_processor=self._data_pre_processor,
                feature_searcher=select_k_best_searcher,
                params_searcher=params_searcher_kneighbors,
                history_manager=ScikitLearnCrossValidationHistoryManager(
                    output_directory='executions_history',
                    models_directory='models_kneighbors',
                    best_params_file_name='kneighbors_best_params',
                    cv_results_file_name='kneighbors_cv_results'
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
            scoring='accuracy',
            history_index=self.history_index,
            stratified=True
        )

    def _execute_additional_validation(self):
        if self._force_execute_additional_validation:
            x, y = self._data_pre_processor.get_data_additional_validation()

            for pipe in self._process_manager.pipelines:
                if self.history_index is not None:
                    history_length = self.history_index + 1
                else:
                    history_length = pipe.history_manager.get_history_len()

                model = pipe.history_manager.get_saved_model(history_length)

                last_history = pipe.history_manager.get_dictionary_from_params_json(history_length - 1)
                selected_features = last_history['features'].replace(' ', '').split(',')
                x_features = x[selected_features]

                additional_validator = ScikitLearnClassifierAdditionalValidator(
                    estimator=model,
                    prefix_file_names=f'{type(model).__name__}_{history_length}',
                    validation_results_directory='additional_validations',
                    data=(x_features, y),
                    show_graphics=False,
                    label_encoder=self._data_pre_processor.tipo_treino_encoder,
                )

                additional_validator.validate()

    def _execute_prediction(self, data_dictionary: dict):
        predict_data = self._data_pre_processor.get_data_to_prediction(data_dictionary)

        last_history = self._process_manager.history_manager.get_dictionary_from_params_json(-1)
        features = last_history['features'].replace(' ', '').split(',')
        predict_data = predict_data[features]

        history_len = self._process_manager.history_manager.get_history_len()
        model = self._process_manager.history_manager.get_saved_model(history_len)

        prediction = model.predict(predict_data)

        return self._data_pre_processor.tipo_treino_encoder.inverse_transform(prediction)