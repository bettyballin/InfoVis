import pandas as pd
from sklearn.manifold import MDS
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


data = pd.read_csv('IVVA2122_H10_Daten.csv', delimiter=';')
dummy = np.array([[1, 2, 3], [2, 3, 4], [0, 1, 2]])

cluster_data = []
for k in ['k3', 'k4', 'k5']:
    cluster_ids = data[k].unique()
    row = []
    for cluster in cluster_ids:
        ids = data[data[k] == cluster]['ID']
        row.append(list(ids))
    cluster_data.append(row)


def inverse_jaccard(set1, set2):
    intersect = len(list(set(set1) & set(set2)))
    union = len(set(set1)) + len(set(set2)) - intersect
    return 1 - intersect / union


jaccard_similarities = []
for i in range(len(cluster_data)):
    for j in range(len(cluster_data[i])):
        cluster_1 = cluster_data[i][j]
        for k in range(len(cluster_data)):
            for l in range(len(cluster_data[k])):
                row = (i, j)
                col = (k, l)
                cluster_2 = cluster_data[k][l]
                inv_jac = inverse_jaccard(cluster_1, cluster_2)
                jaccard_similarities.append((i, j, k, l, inv_jac))

jaccard_array = np.array(pd.DataFrame(jaccard_similarities)[4])
jaccard_matrix = np.array(np.split(jaccard_array, 12))

print()


def ex2(jaccard):
    mds = MDS(dissimilarity="precomputed")
    #pos = mds.fit(jaccard).embedding_  
    print(jaccard.shape)
    transformed = mds.fit_transform(jaccard)
    print(transformed.shape)
    fig = plt.figure()
    plt.scatter(transformed[:,0],transformed[:,1])
    plt.show()

ex2(jaccard_matrix)