import numpy as np
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as shc
import pickle

from scipy.cluster.hierarchy import fclusterdata,fcluster


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
# X = np.load("../surrogate/surrogate_cost_2nets_Cost.npy", allow_pickle=True)
X = np.load("../surrogate/surrogate_cost_2nets_Cost_Haus.npy", allow_pickle=True)

X = X.tolist()
# print(X)
X = [val for key,val in X.items()]
# print("Cost: ",X)
# print("Nets combinations: ",net_combination)

surrogate_loss_dict = {}
for i in range(net_combination.shape[0]):
	surrogate_loss_dict[(net_combination[i,0],net_combination[i,1])] = X[i] #+ np.random.randint(1,100)
	surrogate_loss_dict[(net_combination[i,1],net_combination[i,0])] = X[i] #+ np.random.randint(1,100)

# print("surrogate_loss_dict: ",surrogate_loss_dict)

# cost = np.array(X).reshape(-1,1)
nets = list(set([i[1] for i in surrogate_loss_dict]+[i[0] for i in surrogate_loss_dict]))
nets_name_dict = {}
for i,j in enumerate(nets):
	nets_name_dict[i] = j
print("nets: ",nets)
nets_in_number = [i for i in range(len(nets))]
print("nets_in_number: ",nets_in_number)
print("nets_name_dict: ",nets_name_dict)
nets_in_number = np.array(nets_in_number).reshape(-1,1)

# dend = shc.dendrogram(shc.linkage(nets_in_number, method='ward'))
# fclust1 = fclusterdata(X, 1.0, metric=mydist)
# fclust2 = fclusterdata(X, 1.0, metric='euclidean')

# print(np.allclose(fclust1, fclust2))
# print(fclust2)

l_matrix = shc.linkage(nets_in_number,metric=netdist)
linkage_matrix = shc.dendrogram(shc.linkage(nets_in_number,metric=netdist))


# plt.figure(figsize=(10, 7))  
# plt.title("Dendrograms")  

# print("l_matrix: ",l_matrix)
# print("linkage_matrix: ",linkage_matrix)

# Retrive clusters
max_d = 1.1*1e10
clusters = fcluster(l_matrix, max_d, criterion='distance')
print("Clusters: ", clusters)
# Dump clusters
output_file = "layer_assignment.pkl"

cluster_list = {}; cluster_num_seen = set()
for j,i in enumerate(clusters): 
	if i not in cluster_num_seen: 
		cluster_num_seen.add(i)
		cluster_list[i] = [nets_name_dict[j]]
	else:
		cluster_list[i].append(nets_name_dict[j])
print("cluster_list: ",cluster_list)
with open("results/"+output_file, 'wb') as f:
    pickle.dump(cluster_list, f)

# with open("results/"+output_file, 'rb') as f:
#     cluster_list_check = pickle.load(f)
# print("cluster_list check: ",cluster_list_check)
# print("Debug length check: ",len(clusters)==len(nets))


# plt.axhline(y=6, color='r', linestyle='--')

plt.show()


