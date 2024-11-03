from scikit_learn.hiper_params_search.random_searcher import HalvingRandomCVHipperParamsSearcher
from scikit_learn.history_manager.cross_validator import CrossValidatorHistoryManager
from scikit_learn.process_manager.multi_process_manager import MultiProcessManager
from scikit_learn.process_manager.pipeline import Pipeline
from scikit_learn.validator.cross_validator import CrossValidator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from skopt.learning import RandomForestRegressor, ExtraTreesRegressor
from tabulate import tabulate

from normalize_training_data import get_dataframe_training_data
from sugerir_peso_exercicio.funcoes import get_dataframe_from_cv_results

df = get_dataframe_training_data()

label_encoder = LabelEncoder()
df['exercicio'] = label_encoder.fit_transform(df['exercicio'])

x = df.drop(columns=['repeticoes', 'data'])
y = df['repeticoes']

params_searcher = HalvingRandomCVHipperParamsSearcher(number_candidates='exhaust',
                                                      min_resources=100,
                                                      max_resources=2000,
                                                      resource='n_samples',
                                                      factor=3,
                                                      log_level=1)

cross_validator = CrossValidator(log_level=1)

random_forest_params = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
                     125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220,
                     225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315,
                     320],
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
    'max_features': ['sqrt', 'log2']
}

k_neighbors_params = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}

decision_tree_params = {
    'criterion': ['squared_error', 'absolute_error'],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
    'max_features': ['sqrt', 'log2', None]
}

gradient_boosting_params = {
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
                     125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220,
                     225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315,
                     320],
    'criterion': ['friedman_mse', 'squared_error'],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'max_features': ['sqrt', 'log2']
}

extra_trees_regressor_params = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
                     125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220,
                     225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315,
                     320],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'max_features': ['sqrt', 'log2', None]
}

svr_params = {
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

history_manager_random_forest = CrossValidatorHistoryManager(
    output_directory='random_forest_regressor',
    models_directory='models',
    params_file_name='params',
    cv_results_file_name='cv_results'
)

history_manager_kneighbors = CrossValidatorHistoryManager(
    output_directory='k_neighbors_regressor',
    models_directory='models',
    params_file_name='params',
    cv_results_file_name='cv_results'
)

history_manager_decision_tree = CrossValidatorHistoryManager(
    output_directory='decision_tree_regressor',
    models_directory='models',
    params_file_name='params',
    cv_results_file_name='cv_results'
)

history_manager_gradient_boosting = CrossValidatorHistoryManager(
    output_directory='gradient_boosting_regressor',
    models_directory='models',
    params_file_name='params',
    cv_results_file_name='cv_results'
)

history_manager_extra_trees = CrossValidatorHistoryManager(
    output_directory='extra_trees_regressor',
    models_directory='models',
    params_file_name='params',
    cv_results_file_name='cv_results'
)

pipelines = [
    Pipeline(
        estimator=RandomForestRegressor(),
        params=random_forest_params,
        feature_searcher=None,
        scaler=None,
        params_searcher=params_searcher,
        history_manager=history_manager_random_forest,
        validator=cross_validator
    ),
    Pipeline(
        estimator=KNeighborsRegressor(),
        params=k_neighbors_params,
        feature_searcher=None,
        scaler=None,
        params_searcher=params_searcher,
        history_manager=history_manager_kneighbors,
        validator=cross_validator
    ),
    Pipeline(
        estimator=DecisionTreeRegressor(),
        params=decision_tree_params,
        feature_searcher=None,
        scaler=None,
        params_searcher=params_searcher,
        history_manager=history_manager_decision_tree,
        validator=cross_validator
    ),
    Pipeline(
        estimator=GradientBoostingRegressor(),
        params=gradient_boosting_params,
        feature_searcher=None,
        scaler=None,
        params_searcher=params_searcher,
        history_manager=history_manager_gradient_boosting,
        validator=cross_validator
    ),
    Pipeline(
        estimator=ExtraTreesRegressor(),
        params=extra_trees_regressor_params,
        feature_searcher=None,
        scaler=None,
        params_searcher=params_searcher,
        history_manager=history_manager_extra_trees,
        validator=cross_validator
    )
]

best_params_history_manager = CrossValidatorHistoryManager(output_directory='best_results',
                                                           models_directory='best_models',
                                                           params_file_name='best_params',
                                                           cv_results_file_name='cv_results')
history_index = -1

manager = MultiProcessManager(
    data_x=x,
    data_y=y,
    seed=42,
    fold_splits=10,
    pipelines=pipelines,
    history_manager=best_params_history_manager,
    scoring='neg_mean_squared_error',
    save_history=True,
    history_index=history_index
)

manager.process_pipelines()
#
# print()
# print('Random Forest Results:')
# random_forest_results = history_manager_random_forest.get_dictionary_from_cv_results_json(index=history_index)
# df_random_forest_results = get_dataframe_from_cv_results(random_forest_results, random_forest_params)
# print(tabulate(df_random_forest_results, headers='keys', tablefmt='fancy_grid', showindex=False))
#
# print()
# print('Kneighbors Results:')
# kneighbors_results = history_manager_kneighbors.get_dictionary_from_cv_results_json(index=history_index)
# df_kneighbors_results = get_dataframe_from_cv_results(kneighbors_results, k_neighbors_params)
# print(tabulate(df_kneighbors_results, headers='keys', tablefmt='fancy_grid', showindex=False))
#
# print()
# print('Decision Tree Results:')
# decision_tree_results = history_manager_decision_tree.get_dictionary_from_cv_results_json(index=history_index)
# df_decision_tree_results = get_dataframe_from_cv_results(decision_tree_results, decision_tree_params)
# print(tabulate(df_decision_tree_results, headers='keys', tablefmt='fancy_grid', showindex=False))
#
# print()
# print('Gradient Boosting Results:')
# gradient_boosting_results = history_manager_gradient_boosting.get_dictionary_from_cv_results_json(index=history_index)
# df_gradient_boosting_results = get_dataframe_from_cv_results(gradient_boosting_results, gradient_boosting_params)
# print(tabulate(df_gradient_boosting_results, headers='keys', tablefmt='fancy_grid', showindex=False))
#
# print()
# print('Extra Trees Results:')
# extra_trees_results = history_manager_extra_trees.get_dictionary_from_cv_results_json(index=history_index)
# df_extra_trees_results = get_dataframe_from_cv_results(extra_trees_results, extra_trees_regressor_params)
# print(tabulate(df_extra_trees_results, headers='keys', tablefmt='fancy_grid', showindex=False))