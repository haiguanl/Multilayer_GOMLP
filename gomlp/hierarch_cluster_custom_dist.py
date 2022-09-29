import numpy as np
from scipy.cluster.hierarchy import fclusterdata
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as shc


def mydist(p1, p2):
    diff = p1 - p2
    return np.vdot(diff, diff) ** 0.5

def netdist(netA_ind,netB_ind):
	# print("netA_ind: ",netA_ind)
	netA = nets_name_dict[netA_ind[0]]
	netB = nets_name_dict[netB_ind[0]]
	return surrogate_loss_dict[(netA,netB)]




# X = np.random.randn(100, 1)
net_combination = np.load("../surrogate/surrogate_cost_2nets_NetComb.npy")
X = np.load("../surrogate/surrogate_cost_2nets_Cost.npy", allow_pickle=True)
X = X.tolist()
# print(X)
X = [val for key,val in X.items()]
print("Cost: ",X)
print("Nets combinations: ",net_combination)

surrogate_loss_dict = {}
for i in range(net_combination.shape[0]):
	surrogate_loss_dict[(net_combination[i,0],net_combination[i,1])] = X[i] + np.random.randint(1,100)
	surrogate_loss_dict[(net_combination[i,1],net_combination[i,0])] = X[i] + np.random.randint(1,100)

print("surrogate_loss_dict: ",surrogate_loss_dict)

# cost = np.array(X).reshape(-1,1)
nets = list(set([i[1] for i in surrogate_loss_dict]+[i[0] for i in surrogate_loss_dict]))
nets_name_dict = {}
for i,j in enumerate(nets):
	nets_name_dict[i] = j
print("nets: ",nets)
nets_in_number = [i for i in range(len(nets))]
print("nets_in_number: ",nets_in_number)
nets_in_number = np.array(nets_in_number).reshape(-1,1)

# dend = shc.dendrogram(shc.linkage(nets_in_number, method='ward'))
# fclust1 = fclusterdata(X, 1.0, metric=mydist)
# fclust2 = fclusterdata(X, 1.0, metric='euclidean')

# print(np.allclose(fclust1, fclust2))
# print(fclust2)


plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(nets_in_number,metric=netdist))

plt.axhline(y=6, color='r', linestyle='--')

plt.show()


