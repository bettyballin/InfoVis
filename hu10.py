import pandas as pd
from sklearn.manifold import MDS
import numpy as np
import plotly.express as px
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


data = pd.read_csv('IVVA2122_H10_Daten.csv', delimiter=';')
dummy = np.array([[1, 2, 3], [2, 3, 4], [0, 1, 2]])

cluster_data = []
cluster_metrics = {}

for k in ['k3', 'k4', 'k5']:
    cluster_ids = data[k].unique()
    row = []
    for cluster in cluster_ids:
        shot_put = data[data[k] == cluster]['Shot_Put'].mean()
        m100 = data[data[k] == cluster]['100m'].mean()
        high_jump = data[data[k] == cluster]['High_Jump'].mean()

        ids = data[data[k] == cluster]['ID']
        size = len(ids)
        metrics = {
            'High_Jump': high_jump,
            '100m': m100,
            'Shot_Put': shot_put,
            'Cluster_Size': size
        }
        cluster_metrics[str(k) + ', ' + str(cluster)] = metrics
        row.append(list(ids))

    cluster_data.append(row)

print()


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
    transformed = mds.fit_transform(jaccard)

    print(np.array(cluster_data[0]).shape)
    print(np.array(cluster_data[1]).shape)
    print(np.array(cluster_data[2]).shape)

    fig = plt.figure()
    
    df = pd.DataFrame({"x":transformed[:,0], "y":transformed[:,1], "text":["C3,1","C3,2","C3,3","C4,1","C4,2","C4,3","C4,4","C5,1","C5,2","C5,3","C5,4","C5,5"]})
    fig = px.scatter(df, x="x", y="y", text="text", color="text", size_max=1.2)
    fig.update_traces(textposition='top center')
    fig.update_layout(title_text='k-Means Cluster 10-Kampf', title_x=0.5)
    fig.show()

ex2(jaccard_matrix)
