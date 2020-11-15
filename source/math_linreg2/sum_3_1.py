import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

""" ##############################################################
Preprocessing
###############################################################"""
path_to_data = pl.Path(__file__).parent.joinpath('prostate_poprawione.csv')
df = pd.read_csv(path_to_data)
df = df.drop(['id'], axis=1)
df_train = df[df['train'] == 'T'].drop(['train'], axis=1)
X, y = df_train.values[:, :-1], df_train.values[:, -1]
scaler = StandardScaler()
scaler.fit(df_train)
df_train_scaled = scaler.transform(df_train)
X_scaled, y_scaled = df_train_scaled[:, :-1], df_train_scaled[:, -1]

""" ##############################################################
Zheng Log method implementation (for PCR_ord)
###############################################################"""
def compute_sort_keys_from_tvalues(X):
    if_passed = False
    while if_passed is False:
        try:
            X_pca = PCA(n_components=X.shape[1]).fit_transform(X)
        except np.linalg.LinAlgError:
            pass
        else:
            if_passed = True

    tvalues = sm.OLS(y_scaled, sm.add_constant(X_pca)).fit().tvalues[1:]
    tvalues_square = tvalues * tvalues
    return np.argsort(-tvalues_square)

SORT_KEYS = compute_sort_keys_from_tvalues(X_scaled)

class ZhengLohTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        global SORT_KEYS
        return X[:, SORT_KEYS][:, 0:self.n_components]


""" ##############################################################
Cross validation
###############################################################"""
def perform_cv(model, param_name, param_range):
    cv = KFold(n_splits=10)
    scorer = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error)
    grid = GridSearchCV(model, scoring=scorer, param_grid={param_name: param_range}, cv=cv)
    grid.fit(X_scaled, y_scaled)
    results = pd.DataFrame.from_dict(grid.cv_results_)[['param_' + param_name, 'mean_test_score', 'std_test_score']]
    results['std_test_score'] = results['std_test_score'] / np.sqrt(results.shape[0])
    return results


results_RR = perform_cv(Ridge(), 'alpha', [0.1, 4.3, 11.2, 22.8, 36.6, 74.4, 151.1, 388.8, 1000.0])
results_PCR = perform_cv(Pipeline(steps=[('pca', PCA()), ('linreg', LinearRegression())]), 'pca__n_components', [1, 2, 3, 4, 5, 6, 7, 8])
results_PCRord = perform_cv(Pipeline(steps=[('zlt', ZhengLohTransformer()), ('linreg', LinearRegression())]), 'zlt__n_components', [1, 2, 3, 4, 5, 6, 7, 8])


""" ##############################################################
Converting  "alpha" from RR to the "degrees of freedom".
###############################################################"""
def calculate_degrees_freedom(X, alpha):
    """ function which converts  "alpha" from RR to "degrees of freedom"""
    inner_factor = np.linalg.inv(X.T.__matmul__(X) + alpha * np.identity(X.shape[1]))
    hat_matrix = (X.__matmul__(inner_factor)).__matmul__(X.T)
    return np.trace(hat_matrix)


results_RR['param_alpha'] = results_RR['param_alpha'].apply(lambda x: calculate_degrees_freedom(X_scaled, x))
results_RR = results_RR.rename(columns={'param_alpha': 'param_df'})
results_RR = results_RR.sort_values(by=['param_df'])
results_RR = results_RR.reset_index(drop=True)


""" ##############################################################
Plotting
###############################################################"""
def plot_results(ax, results, param_name, title):
    """ function which creates plot for single model"""
    ax.plot(results[param_name], results['mean_test_score'])
    ax.scatter(results[param_name], results['mean_test_score'])
    ax.errorbar(results[param_name], results['mean_test_score'], results['std_test_score'])
    one_error_above_min = results['mean_test_score'].min() + results['std_test_score'][results['mean_test_score'].argmin()]
    one_error_below_min = results['mean_test_score'].min() - results['std_test_score'][results['mean_test_score'].argmin()]
    ax.hlines(one_error_above_min, 0, 8, linestyles='dashed')
    ax.hlines(one_error_below_min, 0, 8, linestyles='dashed')
    ax.set_xticks(np.linspace(0, 8, 9))
    ax.set_yticks(np.linspace(0, 1.6, 9))
    ax.set_xlabel(param_name)
    ax.set_ylabel('mean squared error')
    ax.set_title(title)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
plot_results(ax1, results_RR, 'param_df', 'ridge')
plot_results(ax2, results_PCR, 'param_pca__n_components', 'pcr')
plot_results(ax3, results_PCRord, 'param_zlt__n_components', 'pcr_ord')

