import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from itertools import cycle
from sklearn.cluster import MeanShift

""" Problem #3 """

df = read_csv('dow_jones_index.csv').drop(columns=['date','stock']).fillna(0.0)
cols_to_format = [
    'open',
    'high',
    'low',
    'close',
    'next_weeks_open',
    'next_weeks_close'
]
for col in cols_to_format:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

clf = MeanShift().fit(df)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
n_clusters_ = len(np.unique(labels))
X = df.values

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(
        X[my_members, 0], # quarter
        X[my_members, 1], # open
        col + '.'
    )
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        'o',
        markerfacecolor=col,
        markeredgecolor='k',
        markersize=14
    )

print("number of estimated clusters : " + str(n_clusters_))
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
