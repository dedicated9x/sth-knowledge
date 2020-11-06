import pathlib as pl
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV

def calculate_degrees_freedom(X, alpha):
    inner_factor = np.linalg.inv(X.T.__matmul__(X) + alpha * np.identity(X.shape[1]))
    hat_matrix = (X.__matmul__(inner_factor)).__matmul__(X.T)
    return np.trace(hat_matrix)


path_to_data = pl.Path(__file__).parent.joinpath('prostate_poprawione.csv')


df = pd.read_csv(path_to_data)
df = df.drop(['id'], axis=1)

df_train = df[df['train'] == 'T'].drop(['train'], axis=1)
X, y = df_train.values[:, :-1], df_train.values[:, -1]

scaler = StandardScaler()
scaler.fit(df_train)

df_train_scaled = scaler.transform(df_train)
X_scaled, y_scaled = df_train_scaled[:, :-1], df_train_scaled[:, -1]


"""just ridge"""
# rgr = Ridge(alpha=36.6)
# rgr.fit(X_scaled, y_scaled)

#TODO ujednolicic

"""####################
RR
####################"""
alpha_range = [0.1, 4.3, 11.2, 22.8, 36.6, 74.4, 151.1, 388.8, 1000.0]
param_grid = dict(alpha=alpha_range)
cv = KFold(n_splits=10)
scorer = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error)
grid = GridSearchCV(Ridge(), scoring=scorer, param_grid=param_grid, cv=cv)

grid.fit(X_scaled, y_scaled)
results = pd.DataFrame.from_dict(grid.cv_results_)[['param_alpha', 'mean_test_score', 'std_test_score']]


"""https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html"""

"""####################
PCA
####################"""
pca = PCA()
linreg = LinearRegression()
pipe = Pipeline(steps=[('pca', pca), ('linreg', linreg)])
param_grid = {
    'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8],
}
scorer = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error)
search = GridSearchCV(pipe, scoring=scorer, param_grid=param_grid, cv=cv)
search.fit(X_scaled, y_scaled)

results2 = pd.DataFrame.from_dict(search.cv_results_)[['param_pca__n_components', 'mean_test_score', 'std_test_score']]


"""####################
PLS
####################"""
n_components_range = [1, 2, 3, 4, 5, 6, 7, 8]
param_grid = dict(n_components=n_components_range)
cv = KFold(n_splits=10)
scorer = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error)
grid = GridSearchCV(PLSRegression(), scoring=scorer, param_grid=param_grid, cv=cv)
grid.fit(X_scaled, y_scaled)

results3 = pd.DataFrame.from_dict(grid.cv_results_)[['param_n_components', 'mean_test_score', 'std_test_score']]





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
z1 = pd.DataFrame.from_dict(grid.cv_results_)
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