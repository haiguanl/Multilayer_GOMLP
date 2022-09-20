import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score

surrogate_results_file = "../surrogate/MLP_ite10000.xlsx"
gomlp_results_file = "../gomlp/results/gomlp_results_50nets.csv"



# surrogate_results = np.loadtxt(surrogate_results_file,delimiter=",",dtype=str)
surrogate_results = pd.read_excel(surrogate_results_file)
surrogate_results_list = surrogate_results.iloc[:,-1].tolist()
print("surrogate_results: ",len(surrogate_results_list))

gomlp_results = pd.read_csv(gomlp_results_file,header=None)
# print("gomlp_results: ",["".join([gomlp_results.iloc[i,j] for j in range(gomlp_results.shape[1])])\
#  for i in range(gomlp_results.shape[0])])

# print("gomlp_results: ",gomlp_results.shape)
gomlp_results_str = ["".join(str(ite) for ite in gomlp_results.iloc[i,:].tolist()[:-1])
 for i in range(gomlp_results.shape[0])]
gomlp_results = [float(i) for i in gomlp_results_str]
print("gomlp_results: ",len(gomlp_results))



# plot scatter
# print("R2: ",r2_score(surrogate_results_list,gomlp_results))
plt.scatter(gomlp_results,surrogate_results_list,color="r",s=10)
plt.xlabel("GOMLP")
plt.ylabel("Surrogate: MLP")
plt.show()


