import pandas as pd
from sklearn.manifold import MDS
import numpy as np
import plotly.graph_objects as go
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

print(cluster_metrics)


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
    
    fig = plt.figure()
    c = cluster_metrics
    df = pd.DataFrame({"x":transformed[:,0], "y":transformed[:,1], "len":[k["Cluster_Size"] for k in c.values()], "text":[t[3:] for t in c.keys()]})
    fig = px.scatter(df, x="x", y="y", color=[t[:2] for t in c.keys()], size="len", text="text",color_discrete_sequence=['blue', 'orange', 'purple'], size_max=100)
    fig.update_traces(textposition='bottom center')
    fig.update_layout(
        legend=dict(
        x=.95,
        y=.95,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    ))
    #/np.sum(np.array([float(k["High_Jump"]) 
    highjump = [float(k["High_Jump"])/10 for k in c.values()]  
    m100 = [float(k["100m"])/10 for k in c.values()]
    shotput = [float(k["Shot_Put"])/10 for k in c.values()]

    fig.add_trace(go.Bar(x=[x-0.001 for x in df["x"]],y=highjump, width=0.001,text=[round(float(k["High_Jump"]),4) for k in c.values()],marker={"color":"yellow"},base=list([-1.5]*12)))
    fig.add_trace(go.Bar(x=df["x"],y=m100, width=0.001,text=[round(float(k["100m"]),4) for k in c.values()],marker={"color":"brown"},base=list([-1.5]*12)))
    fig.add_trace(go.Bar(x=[x+0.001 for x in df["x"]],y=shotput, width=0.001,text=[round(float(k["Shot_Put"]),4) for k in c.values()],marker={"color":"pink"},base=list([-1.5]*12)))

    fig.update_layout(title_text='k-Means Cluster 10-Kampf', title_x=0.5)
    fig.show()

ex2(jaccard_matrix)
