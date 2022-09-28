import numpy as np
from scipy.cluster.hierarchy import fclusterdata
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as shc


def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5

def netdist(netA,netB):
	return surrogate_loss_dict[(netA,neB)]




X = np.random.randn(100, 2)


dend = shc.dendrogram(shc.linkage(X, method='ward'))
fclust1 = fclusterdata(X, 1.0, metric=mydist)
fclust2 = fclusterdata(X, 1.0, metric='euclidean')

# print(np.allclose(fclust1, fclust2))
print(fclust2)


plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X,metric=mydist))
plt.axhline(y=6, color='r', linestyle='--')
plt.show()