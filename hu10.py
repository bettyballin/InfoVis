import pandas as pd
from sklearn.manifold import MDS
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

def ex2(jaccard):
    mds = MDS(dissimilarity="precomputed")
    #pos = mds.fit(jaccard).embedding_  
    
    transformed = mds.fit_transform(jaccard)
    print(transformed)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(transformed[0],transformed[1])
    plt.show()


df = pd.read_csv('IVVA2122_H10_Daten1.csv', delimiter=";")
x = np.zeros([2,3476])
x[0,:] = df["k4"]
x[1,:] = df["k5"]
ex2(x)