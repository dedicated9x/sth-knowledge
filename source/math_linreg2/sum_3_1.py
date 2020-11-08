
import pathlib as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV

""" data preprocessing """
path_to_data = pl.Path(__file__).parent.joinpath('prostate_poprawione.csv')
df = pd.read_csv(path_to_data)
df = df.drop(['id'], axis=1)
df_train = df[df['train'] == 'T'].drop(['train'], axis=1)
X, y = df_train.values[:, :-1], df_train.values[:, -1]
scaler = StandardScaler()
scaler.fit(df_train)
df_train_scaled = scaler.transform(df_train)
X_scaled, y_scaled = df_train_scaled[:, :-1], df_train_scaled[:, -1]


cv = KFold(n_splits=10)
scorer = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error)

"""####################
RR
####################"""
model_1 = Ridge()
param_name_1 = 'alpha'
param_range_1 = [0.1, 4.3, 11.2, 22.8, 36.6, 74.4, 151.1, 388.8, 1000.0]

grid1 = GridSearchCV(model_1, scoring=scorer, param_grid={param_name_1: param_range_1}, cv=cv)
grid1.fit(X_scaled, y_scaled)
results1 = pd.DataFrame.from_dict(grid1.cv_results_)[['param_' + param_name_1, 'mean_test_score', 'std_test_score']]


"""####################
PCA
####################"""
model_2 = Pipeline(steps=[('pca', PCA()), ('linreg', LinearRegression())])
param_name_2 = 'pca__n_components'
param_range_2 = [1, 2, 3, 4, 5, 6, 7, 8]

grid2 = GridSearchCV(model_2, scoring=scorer, param_grid={param_name_2: param_range_2}, cv=cv)
grid2.fit(X_scaled, y_scaled)
results2 = pd.DataFrame.from_dict(grid2.cv_results_)[['param_' + param_name_2, 'mean_test_score', 'std_test_score']]


"""####################
PLS
####################"""
model_3 = PLSRegression()
param_name_3 = 'n_components'
param_range_3 = [1, 2, 3, 4, 5, 6, 7, 8]

grid3 = GridSearchCV(model_3, scoring=scorer, param_grid={param_name_3: param_range_3}, cv=cv)
grid3.fit(X_scaled, y_scaled)
results3 = pd.DataFrame.from_dict(grid3.cv_results_)[['param_' + param_name_3, 'mean_test_score', 'std_test_score']]


""" converts "alpha" from RR to "degrees of freedom"""
def calculate_degrees_freedom(X, alpha):
    inner_factor = np.linalg.inv(X.T.__matmul__(X) + alpha * np.identity(X.shape[1]))
    hat_matrix = (X.__matmul__(inner_factor)).__matmul__(X.T)
    return np.trace(hat_matrix)

results1['param_alpha'] = results1['param_alpha'].apply(lambda x: calculate_degrees_freedom(X_scaled, x))
results1 = results1.rename(columns={'param_alpha': 'param_df'})
results1 = results1.sort_values(by=['param_df'])
results1 = results1.reset_index(drop=True)


""" function which create plot for single model"""
def plot_results(ax, results, param_name, title):
    ax.plot(results[param_name], results['mean_test_score'])
    ax.scatter(results[param_name], results['mean_test_score'])
    ax.errorbar(results[param_name], results['mean_test_score'], results['std_test_score'])
    one_error_above_min = results['mean_test_score'].min() + results['std_test_score'][results['mean_test_score'].argmin()]
    ax.hlines(one_error_above_min, 0, 8)
    ax.set_xticks(np.linspace(0, 8, 9))
    ax.set_title(title)

# TODO wpisac to do
""" creating multiple plots """
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
plot_results(ax1, results1, 'param_df', 'ridge')
plot_results(ax2, results2, 'param_pca__n_components', 'pcr')
plot_results(ax3, results3, 'param_n_components', 'pls')




""" linreg wytrenowany na PCA od razu (!) predyktuje dobrze"""
# pca_best = PCA(n_components=8)
# pca_best.fit(X_scaled)
# X_pca = pca_best.transform(X_scaled)
# rgr = LinearRegression()
# rgr.fit(X_pca, y_scaled)


"""test"""
# rgr.score(X_scaled, y_scaled)
# rgr.score(X_pca, y_scaled)
# z1 = pd.DataFrame().assign(
#     rec=rgr.predict(X_scaled),
#     exp=y_scaled
# )




""" test error to pstd"""
import statistics
import math
# z1 = pd.DataFrame.from_dict(grid.cv_results_)
# z2 = z1.iloc[:, 6:11]
# for i in range(9):
#     print(i, math.sqrt(statistics.pvariance(z2.iloc[i, :])))



""" prediction (squared) error"""
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
sklearn.metrics.mean_squared_error(y_true, y_pred)



""" standard error (of mean)"""
some_errors = [2.1, 2.4, 1.8, 1.7, 2.5]
scipy.stats.sem(some_errors)


""" przy malych nie ma roznicy"""
# for i in [0.1, 1, 10 , 100, 1000, 10000]:
#     print(calculate_degrees_freedom(X, i), calculate_degrees_freedom(X_scaled, i))


""" sprawdzenie alpha_range"""
# for i in alpha_range:
#     print(i, calculate_degrees_freedom(X_scaled, i))