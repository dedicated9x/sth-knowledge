import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

var_e = 0.2
var_h = 0.4

mean_h = [0, 0]
mean_e1 = [-1, 1]
mean_e2 = [1, 1]

p = 0.5



pop = []
for i in range(10000):
    mean, var = random.choices(
        population=[(mean_h, var_h), (mean_e1, var_e), (mean_e2, var_e)],
        weights=[p, (1 - p) / 2, (1 - p) / 2]
    )[0]
    pop.append(np.random.multivariate_normal(mean, (var ** 2) * np.identity(2)))


z1 = np.array(pop, dtype=np.dtype('float'))
plt.scatter(z1[:, 0], z1[:, 1])

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(pop)
labels = kmeans.labels_
df = pd.DataFrame(data=pop, columns=['X1', 'X2']).assign(cluster=kmeans.labels_)


plt.scatter(df[df['cluster'] == 1]['X1'], df[df['cluster'] == 1]['X2'])
plt.scatter(df[df['cluster'] == 2]['X1'], df[df['cluster'] == 2]['X2'])
plt.scatter(df[df['cluster'] == 0]['X1'], df[df['cluster'] == 0]['X2'])
