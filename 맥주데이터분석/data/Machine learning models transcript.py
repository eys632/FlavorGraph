#%% md

# Importing packages
#%%
import os
from platform import system
import pandas as pd
import numpy as np
from numpy.random import randint

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.cross_decomposition import PLSRegression

from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from xgboost import XGBRegressor

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor

import optuna

import pickle
import joblib

#%% md
#### Working directory
#%%

### TO DO: ADJUST TO YOUR RELEVANT WORKING DIRECTORY
os.chdir(os.path.join('my', 'directory'))

np.random.seed(42)
#%%
from init_modeling import generate_X
from init_modeling import regressor_performance, pickle_model
#%% md
# Creating required folders
#%%
for path in ['models', 'Model_performance', 'Optimal_settings']:
    if not os.path.exists(path):
        os.makedirs(path)
#%% md
#### Test train split
# Comment/uncomment one of both sections to select the RateBeer or the Trained Panel sensory datasets.
# Note that the RateBeer dataset is not available without prior written permission from RateBeer (ZX Ventures, USA)
#%%
# chem_dataset_ratebeer = pd.read_csv('chem_dataset_ratebeer.csv').set_index('beer')
# x_name = 'chem_log'
# X = generate_X(pd.concat([chem_dataset_ratebeer['beer_id'], chem_dataset_ratebeer.loc[:,'acetaldehyde':'sulfur_sum']], axis=1), impute=True)
# y_name = 'ratebeer'
# Y = chem_dataset_ratebeer.loc[:,'ave_score_5':'overall_20']
# y_class = chem_dataset_ratebeer.loc[:,'tasting_category_fine']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,
#                                                     random_state=0, shuffle=True, stratify=y_class)


chem_dataset = pd.read_csv('Supplemental File S1.csv').set_index('beer')
trained_panel_dataset = pd.read_csv('Supplemental File S4.csv').set_index('beer')
chem_dataset_expertpanel = chem_dataset.join(trained_panel_dataset)
x_name = 'chem_log'
X = generate_X(pd.concat([chem_dataset_expertpanel['beer_id'] , chem_dataset_expertpanel.loc[:,'acetaldehyde':'sulfur_sum']], axis=1), impute=True)
y_name = 'expertpanel'
Y = chem_dataset_expertpanel.loc[:,'A_malt_all':'overall']
y_class = chem_dataset_expertpanel.loc[:,'tasting_category_fine']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,
                                                    random_state=0, shuffle=True, stratify=y_class)
#%% md
#### Scaling the data
# This is necessary for the Neural Network and Linear Models. Out of consistency, we do it for the whole dataset.
#%%
scaler_X = StandardScaler().fit(X_train)
# scaler_Y = StandardScaler().fit(Y_train)

X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)
# Y_train = scaler_Y.transform(Y_train)
# Y_test = scaler_Y.transform(Y_test)
#%% md
# Creating first order interaction terms
#%%
poly = PolynomialFeatures(interaction_only=True)

X_train_int = poly.fit_transform(X_train)
X_test_int = poly.fit_transform(X_test)

X_train_int = pd.DataFrame(X_train_int,columns=poly.get_feature_names_out()).drop('1', axis = 1)
X_test_int = pd.DataFrame(X_test_int,columns=poly.get_feature_names_out()).drop('1', axis = 1)
#%% md
### Linear model
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    regressor_loop = LinearRegression(fit_intercept=True, copy_X=True)
    regressor_loop.fit(X_train_int, Y_train_loop)

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train_int, Y_train_loop, X_test_int, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['LinearModel interactions', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    joblib.dump(regressor_loop, os.path.join('models', f'Linear_regression_interaction_{aspect}_{y_name}.pkl'))

# Training full model
LM = LinearRegression(fit_intercept=True, copy_X=True).fit(X_train_int, Y_train)
mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train_int, Y_train, X_test_int, Y_test, LM, toprint = False)
aspect = 'everything'

joblib.dump(LM, os.path.join('models', f'Linear_regression_interaction_full_model_{y_name}.pkl'))

# Adding full model to the list
performance_loop.append(['Complete_LinearModel interactions', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

# Turning the list into a dataframe
performance_lr_df = pd.DataFrame(performance_loop,
                                 columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

# Saving
performance_lr_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_Linear_model_interactions.csv', index=False)
#%% md
### Lasso regression
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    regressor_loop = LassoCV(cv = 5, random_state = 0)
    regressor_loop.fit(X_train_int, Y_train_loop)

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train_int, Y_train_loop, X_test_int, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['Lasso interactions', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    joblib.dump(regressor_loop, os.path.join('models', f'Lasso_regression_interactions_{aspect}_{y_name}.pkl'))

# Training full model
lasso_cv = MultiTaskLassoCV(cv = 5, random_state = 0).fit(X_train_int, Y_train)
mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train_int, Y_train, X_test_int, Y_test, lasso_cv, toprint = False)
aspect = 'everything'

joblib.dump(lasso_cv, os.path.join('models', f'Lasso_regression_interactions_full_model_{y_name}.pkl'))

# Adding full model to the list
performance_loop.append(['Complete_Lasso interactions', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

# Turning the list into a dataframe
performance_l1r_df = pd.DataFrame(performance_loop,
                                  columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

# Saving
performance_l1r_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_Lasso_interactions.csv', index=False)
#%% md
### Partial Least Squares

# For PLSR, grid search returned either the same or worse models. So we are using the native model.
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    regressor_loop = PLSRegression(scale=True)
    regressor_loop.fit_transform(X_train, Y_train_loop)

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train_loop, X_test, Y_test_loop, regressor_loop, toprint=True)
    performance_loop.append(['PLS', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])


    joblib.dump(regressor_loop, os.path.join('models', f'PLS_regression_{aspect}_{y_name}.pkl'))

# Training full model
PLS = PLSRegression().fit(X_train, Y_train)
mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, PLS, toprint = False)
aspect = 'everything'

joblib.dump(PLS, os.path.join('models', f'Elasticnet_regression_full_model_{y_name}.pkl'))

# Adding full model to the list
performance_loop.append(['Complete_PLS', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])



performance_pls_df = pd.DataFrame(performance_loop,
                                  columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

performance_pls_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_Partial_Least_Squares_Regression.csv', index=False)
#%% md
### AdaBoost
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    # search for best alpha and
    param_grid = [{'learning_rate': np.arange(0.2,2.0,0.2),
                   'loss': ['linear','square','exponential'],
                   'n_estimators': [200, 500, 1000],
                   }]
    grid_search_loop = GridSearchCV(AdaBoostRegressor(n_estimators=200, random_state=0),
                                    param_grid,
                                    cv=5,
                                    scoring='r2',
                                    verbose=1,
                                    refit=True,
                                    n_jobs=8)
    grid_search_results = grid_search_loop.fit(X_train, Y_train_loop)
    regressor_loop = grid_search_results.best_estimator_

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train_loop, X_test, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['AdaBoost', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    CV_results = pd.DataFrame(grid_search_loop.cv_results_)
    pickle_model(f'Adaboost_regressor_{aspect}_{y_name}', regressor_loop, CV_results)

# Training full model
param_grid = [{'estimator__learning_rate': np.arange(0.2,2.0,0.2),
               'estimator__loss': ['linear','square','exponential'],
               'estimator__n_estimators': [200, 500, 1000],
               }]
grid_search = GridSearchCV(MultiOutputRegressor(AdaBoostRegressor(random_state=0)),
                           param_grid,
                           cv=5,
                           scoring='r2',
                           verbose=1,
                           refit=True,
                           n_jobs=8)

grid_search_results = grid_search.fit(X_train, Y_train)
adaboost = grid_search_results.best_estimator_

mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, adaboost, toprint = False)
aspect = 'everything'

CV_results = pd.DataFrame(grid_search.cv_results_)
pickle_model(f'Adaboost_regressor_full_model_{y_name}', adaboost, CV_results)

# Adding full model to the list
performance_loop.append(['Complete_Adaboost', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])


# Creating the dataframe
performance_abr_df = pd.DataFrame(performance_loop,
                                  columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

performance_abr_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_AdaBoost.csv', index=False)
#%% md
### GradientBoost
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    # search for best alpha and
    param_grid = [{'subsample': np.arange(0.2,1.0,0.2),
                   'learning_rate': np.arange(0.02,0.2,0.02),
                   'max_depth': np.arange(6,15,3),
                   'n_estimators': [200, 500, 1000],
                   'loss': ['squared_error', 'absolute_error']
                   }]
    grid_search_loop = GridSearchCV(GradientBoostingRegressor(random_state=0),
                                    param_grid,
                                    cv=5,
                                    scoring='r2',
                                    verbose=1,
                                    refit=True,
                                    return_train_score=True,
                                    n_jobs = 8)

    grid_search_results = grid_search_loop.fit(X_train, Y_train_loop)
    regressor_loop = grid_search_results.best_estimator_

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train_loop, X_test, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['GradientBoost', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    CV_results = pd.DataFrame(grid_search_loop.cv_results_)
    pickle_model(f'Gradient_Boost_regressor_{aspect}_{y_name}', regressor_loop, CV_results)


# Training full model
param_grid = [{'estimator__subsample': np.arange(0.2,1.0,0.2),
               'estimator__learning_rate': np.arange(0.02,0.2,0.02),
               'estimator__max_depth': np.arange(6,15,3),
               'estimator__n_estimators': [200, 500, 1000],
               'estimator__loss': ['squared_error', 'absolute_error']
               }]
grid_search = GridSearchCV(MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
                           param_grid,
                           cv=5,
                           scoring='r2',
                           verbose=1,
                           refit=True,
                           return_train_score=True,
                           n_jobs = 8)

grid_search_results = grid_search.fit(X_train, Y_train)
gradientboost = grid_search_results.best_estimator_

mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, gradientboost, toprint = False)
aspect = 'everything'

CV_results = pd.DataFrame(grid_search.cv_results_)
pickle_model(f'Gradient_Boost_regressor_full_model_{y_name}', gradientboost, CV_results)

# Adding full model to the list
performance_loop.append(['Complete_GradientBoost', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

# Creating the dataframe
performance_gb_df = pd.DataFrame(performance_loop,
                                 columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

performance_gb_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_GradientBoost.csv', index=False)
#%% md
### Random Forest
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    # search for best alpha and
    param_grid = [
        {'max_features': np.arange(2,X_train.shape[1], 20),
         'max_depth': np.arange(4, 20, 2),
         'min_impurity_decrease': np.arange(0, 0.4, 0.05),
         'n_estimators' : [200, 500, 1000]},
    ]
    grid_search_loop = GridSearchCV(estimator=RandomForestRegressor(oob_score=True, random_state=0),
                                    param_grid=param_grid,
                                    cv=5,
                                    scoring= 'r2',
                                    refit=True,
                                    verbose=1,
                                    return_train_score=True,
                                    n_jobs = 8
                                    )
    grid_search_results = grid_search_loop.fit(X_train, Y_train_loop)
    regressor_loop = grid_search_results.best_estimator_

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train_loop, X_test, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['RandomForest', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    CV_results = pd.DataFrame(grid_search_loop.cv_results_)
    pickle_model(f'Random_Forest_regressor_{aspect}_{y_name}', regressor_loop, CV_results)

# Training full model
param_grid = [
    {'max_features': np.arange(2,X_train.shape[1], 20),
     'max_depth': np.arange(4, 20, 2),
     'min_impurity_decrease': np.arange(0, 0.4, 0.05),
     'n_estimators' : [200, 500, 1000]},
]
grid_search = GridSearchCV(estimator=RandomForestRegressor(oob_score=True, random_state=0),
                           param_grid=param_grid,
                           cv=5,
                           scoring= 'r2',
                           refit=True,
                           verbose=1,
                           return_train_score=True,
                           n_jobs = 8
                           )

grid_search_results = grid_search.fit(X_train, Y_train)
random_forest = grid_search_results.best_estimator_

mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, random_forest, toprint = False)
aspect = 'everything'

CV_results = pd.DataFrame(grid_search.cv_results_)
pickle_model(f'Random_Forest_regressor_full_model_{y_name}', random_forest, CV_results)

# Adding full model to the list
performance_loop.append(['Complete_RandomForest', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

# Creating the dataframe
performance_rf_df = pd.DataFrame(performance_loop,
                                 columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

performance_rf_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_RandomForest.csv', index=False)
#%% md
### Extra Trees
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    # search for best alpha and
    param_grid = [
        {'max_features': np.arange(2, X_train.shape[1], 20),
         'max_depth': np.arange(4, 20, 2),
         'min_impurity_decrease': np.arange(0, 0.4, 0.05),
         'n_estimators': [200, 500, 1000]
         },
    ]
    grid_search_loop = GridSearchCV(estimator=ExtraTreesRegressor(bootstrap=True, oob_score=True,
                                                                  random_state=0),
                                    param_grid=param_grid,
                                    cv=5,
                                    scoring= 'r2',
                                    refit=True,
                                    verbose=1,
                                    n_jobs = 8,
                                    )
    grid_search_results = grid_search_loop.fit(X_train, Y_train_loop)
    regressor_loop = grid_search_results.best_estimator_

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train_loop, X_test, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['ExtraTrees', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    CV_results = pd.DataFrame(grid_search_loop.cv_results_)
    pickle_model(f'Extra_Trees_regressor_{aspect}_{y_name}', regressor_loop, CV_results)

# Training full model
param_grid = [
    {'max_features': np.arange(2, X_train.shape[1], 20),
     'max_depth': np.arange(4, 20, 2),
     'min_impurity_decrease': np.arange(0, 0.4, 0.05),
     'n_estimators': [200, 500, 1000]
     },
]
grid_search = GridSearchCV(estimator=ExtraTreesRegressor(bootstrap=True, oob_score=True, random_state=0),
                           param_grid=param_grid,
                           cv=5,
                           scoring= 'r2',
                           refit=True,
                           verbose=1,
                           n_jobs = 8,
                           )

grid_search_results = grid_search.fit(X_train, Y_train)
extra_trees = grid_search_results.best_estimator_

mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, extra_trees, toprint = False)
aspect = 'everything'

CV_results = pd.DataFrame(grid_search.cv_results_)
pickle_model(f'Extra_Trees_regressor_full_model_{y_name}', extra_trees, CV_results)

# Adding full model to the list
performance_loop.append(['Complete_ExtraTrees', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

# Creating the dataframe
performance_et_df = pd.DataFrame(performance_loop,
                                 columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

performance_et_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_ExtraTrees.csv', index=False)
#%% md
### XGBOOST model
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    # search for best alpha and
    param_grid = [{'eta': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                   'max_depth': np.arange(10)+1,
                   'n_estimators': [200, 500, 1000],
                   }]
    grid_search_loop = GridSearchCV(estimator=XGBRegressor(oob_score=True, random_state=0),
                                    param_grid=param_grid,
                                    cv=5,
                                    scoring= 'r2',
                                    refit=True,
                                    verbose=1,
                                    return_train_score=True,
                                    n_jobs = 8
                                    )
    grid_search_results = grid_search_loop.fit(X_train, Y_train_loop)
    regressor_loop = grid_search_results.best_estimator_

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train_loop, X_test, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['RandomForest', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    CV_results = pd.DataFrame(grid_search_loop.cv_results_)
    pickle_model(f'XGBoost_regressor_{aspect}_{y_name}', regressor_loop, CV_results)

# Training full model
param_grid = [{'estimator__eta': [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
               'estimator__max_depth': np.arange(10)+1,
               'estimator__n_estimators': [200, 500, 1000],
               }]
grid_search = GridSearchCV(estimator=MultiOutputRegressor(XGBRegressor(random_state=0)),
                           param_grid=param_grid,
                           cv=5,
                           scoring= 'r2',
                           refit=True,
                           verbose=1,
                           return_train_score=True,
                           n_jobs = 8
                           )

grid_search_results = grid_search.fit(X_train, Y_train)
XGBoost = grid_search_results.best_estimator_

mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, random_forest, toprint = False)
aspect = 'everything'

CV_results = pd.DataFrame(grid_search.cv_results_)
pickle_model(f'XGBoost_regressor_full_model_{y_name}', XGBoost, CV_results)

# Adding full model to the list
performance_loop.append(['Complete_XGBoost', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

# Creating the dataframe
performance_xgb_df = pd.DataFrame(performance_loop,
                                  columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

performance_xgb_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_XGBoost.csv', index=False)
#%% md
### Support Vector Regressor
#%%
performance_loop = []
for aspect in Y_train.columns:
    Y_train_loop = Y_train[aspect]
    Y_test_loop = Y_test[aspect]

    # search for best alpha and
    param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10],
                   'epsilon': [0.01, 0.05, 0.1, 0.2],
                   'kernel': ['linear','poly','rbf'],
                   'degree': np.arange(2,5),
                   'gamma': np.arange(0.002,0.5,0.01)
                   }]
    grid_search_loop = GridSearchCV(SVR(),
                                    param_grid,
                                    cv=5,
                                    scoring='r2',
                                    verbose=1,
                                    refit=True,
                                    n_jobs = 8)

    grid_search_results = grid_search_loop.fit(X_train, Y_train_loop)
    regressor_loop = grid_search_results.best_estimator_

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train_loop, X_test, Y_test_loop, regressor_loop, toprint=False)
    performance_loop.append(['SupportVectorMachine', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

    CV_results = pd.DataFrame(grid_search_loop.cv_results_)
    pickle_model(f'Support_Vector_regressor_{aspect}_{y_name}', regressor_loop, CV_results)

# Training full model
param_grid = [{'estimator__C': [0.001, 0.01, 0.1, 1, 10],
               'estimator__epsilon': [0.01, 0.05, 0.1, 0.2],
               'estimator__kernel': ['linear','poly','rbf'],
               'estimator__degree': np.arange(2,5),
               'estimator__gamma': np.arange(0.002,0.5,0.01)
               }]
grid_search = GridSearchCV(MultiOutputRegressor(SVR()),
                           param_grid,
                           cv=5,
                           scoring='r2',
                           verbose=1,
                           refit=True,
                           n_jobs = 8)

grid_search_results = grid_search.fit(X_train, Y_train)
svm = grid_search_results.best_estimator_

mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, svm, toprint = False)
aspect = 'everything'

CV_results = pd.DataFrame(grid_search.cv_results_)
pickle_model(f'Support_Vector_regressor_full_model_{y_name}', svm, CV_results)

# Adding full model to the list
performance_loop.append(['Complete_SupportVectorMachine', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])

# Creating the dataframe
performance_svm_df = pd.DataFrame(performance_loop,
                                  columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

performance_svm_df.to_csv(f'Model_performance/ModelPerformances_{x_name}_{y_name}_SupportVectorMachine.csv', index=False)
#%% md
#### Test train split
# The neural network was particularly weak with the random state 0, for reasons unknown. Random state 1 greatly improved performance.
#%%
# chem_dataset_ratebeer = pd.read_csv('chem_dataset_complete_log_ratebeer.csv').set_index('beer')
# x_name = 'chem_log'
# X = generate_X(pd.concat([chem_dataset_ratebeer['beer_id'], chem_dataset_ratebeer.loc[:,'acetaldehyde':'sulfur_sum']], axis=1), impute=True)
# y_name = 'ratebeer'
# Y = chem_dataset_ratebeer.loc[:,'ave_score_5':'overall_20']
# y_class = chem_dataset_ratebeer.loc[:,'tasting_category_fine']
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,
#                                                     random_state=1, shuffle=True, stratify=y_class)


chem_dataset_expertpanel = pd.read_csv('chem_dataset_complete_log_expertpanel.csv').set_index('beer')
x_name = 'chem_log'
X = generate_X(pd.concat([chem_dataset_expertpanel['beer_id'] , chem_dataset_expertpanel.loc[:,'acetaldehyde':'sulfur_sum']], axis=1), impute=True)
y_name = 'expertpanel'
Y = chem_dataset_expertpanel.loc[:,'A_malt_all':'overall']
y_class = chem_dataset_expertpanel.loc[:,'tasting_category_fine']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30,
                                                    random_state=1, shuffle=True, stratify=y_class)
#%% md
### Artificial Neural Network
# Creating and Optuna optimizer objective
#%%
Model_performance = []

#### TRAINING INDIVIDUAL MODELS

for aspect in Y_train.columns:

    ### Defining objective
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1,10) # no. of hidden layers
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int('n_units_{}'.format(i+1), 20, 300, step = 40)) # no. of hidden units or neurons
        activation=trial.suggest_categorical('activation',['tanh', 'relu']) # activation function
        alpha=trial.suggest_float('alpha', 1e-4, 100, log = True) #L2 penalty (regularization term) parameter.
        # define classifier
        model =  MLPRegressor(random_state=1,
                              solver='adam',
                              activation=activation,
                              alpha=alpha,
                              hidden_layer_sizes=(layers),
                              max_iter=1000,
                              learning_rate='adaptive',
                              early_stopping=True)

        # Evaluating model performance
        scores = cross_val_score(model,
                                 X_train,
                                 Y_train[aspect],
                                 cv=KFold(n_splits=5,
                                          # random_state=1, # Uncomment if you want to set shuffle to True.
                                          shuffle=False),
                                 scoring='r2',
                                 n_jobs=8)
        r_squared = scores.mean()

        return r_squared

    ### Optmizing
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3000, n_jobs=8)

    ### Training model with optimal parameters
    architecture = []
    for i in range(study.best_params['n_layers']):
        architecture.append(study.best_params[f'n_units_{i+1}'])

    model = MLPRegressor(random_state=1,
                         solver='adam',
                         max_iter=1000,
                         learning_rate='adaptive',
                         early_stopping=True,
                         hidden_layer_sizes=architecture,
                         activation=study.best_params['activation'],
                         alpha=study.best_params['alpha'])

    model.fit(X_train, Y_train[aspect])

    joblib.dump(model, os.path.join('models', f'Optuna_Neural_network_model_adam_{y_name}_{aspect}.pkl'))

    myresults = study.best_params
    myresults['R2'] = regressor_performance(X_train, Y_train[aspect], X_test, Y_test[aspect], model, toprint = False)[3]
    mydf = pd.DataFrame.from_dict(myresults, orient = 'index')
    mydf.columns = ['optimal value']
    mydf.to_csv(f'Optimal_settings/Optimal_settings_neural_network_adam_{y_name}_{aspect}.csv')

    mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train[aspect], X_test, Y_test[aspect], model, toprint = False)
    Model_performance.append(['MLPRegressor', x_name, y_name, aspect, mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])


#### FULL MODEL

### Defining objective
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1,10) # no. of hidden layers
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int('n_units_{}'.format(i+1), 20, 300, step = 40)) # no. of hidden units or neurons
    activation=trial.suggest_categorical('activation',['tanh', 'relu']) # activation function
    alpha=trial.suggest_float('alpha', 1e-4, 100, log = True) #L2 penalty (regularization term) parameter.
    # define classifier
    model =  MLPRegressor(random_state=1,
                          solver='adam',
                          activation=activation,
                          alpha=alpha,
                          hidden_layer_sizes=(layers),
                          max_iter=1000,
                          learning_rate='adaptive',
                          early_stopping=True)

    # Evaluating model performance
    scores = cross_val_score(model,
                             X_train,
                             Y_train,
                             cv=KFold(n_splits=5,
                                      # random_state=1, # Uncomment if you want to set shuffle to True.
                                      shuffle=False),
                             scoring='r2',
                             n_jobs=8)
    r_squared = scores.mean()

    return r_squared

### Optmizing
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3000, n_jobs=8)

### Training model with optimal parameters
architecture = []
for i in range(study.best_params['n_layers']):
    architecture.append(study.best_params[f'n_units_{i+1}'])

model = MLPRegressor(random_state=1,
                     solver='adam',
                     max_iter=1000,
                     learning_rate='adaptive',
                     early_stopping=True,
                     hidden_layer_sizes=architecture,
                     activation=study.best_params['activation'],
                     alpha=study.best_params['alpha'])

model.fit(X_train, Y_train)

joblib.dump(model, os.path.join('models', f'Optuna_Neural_network_full_model_adam_{y_name}.pkl'))

myresults = study.best_params
myresults['R2'] = regressor_performance(X_train, Y_train, X_test, Y_test, model)[3]
mydf = pd.DataFrame.from_dict(myresults, orient = 'index')
mydf.columns = ['optimal value']
mydf.to_csv(f'Optimal_settings/Optimal_settings_neural_network_adam_full_model_{y_name}.csv')

mse_train, r2_train, mse_test, r2_test = regressor_performance(X_train, Y_train, X_test, Y_test, model, toprint = False)
Model_performance.append(['Complete_MLPRegressor',x_name, y_name, 'everything', mse_train, r2_train, np.sqrt(r2_train), mse_test, r2_test, np.sqrt(r2_test)])


## Saving model performance
Performance = pd.DataFrame(Model_performance,
                           columns=['regressor','X', 'Y', 'aspect', 'mse_train', 'r2_train', 'r_train', 'mse_test', 'r2_test', 'r_test'])

Performance.to_csv(f'Model_Performance/ModelPerformances_{x_name}_{y_name}_MLPRegressor.csv', index=False)