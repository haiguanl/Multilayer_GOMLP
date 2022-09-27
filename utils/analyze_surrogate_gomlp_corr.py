import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats 

surrogate_results_file = "../surrogate/Tree.xlsx"
# surrogate_results_file = "../surrogate/MLP_ite10000.xlsx"

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



# print("R2: ",r2_score(surrogate_results_list,gomlp_results))
# r = np.corrcoef(np.array(gomlp_results).reshape(-1,1),np.array(surrogate_results_list).reshape(-1,1))
# print("R score: ",r[0,1])
# correlation study

slope, intercept, r, p, std_err = stats.linregress(surrogate_results_list, gomlp_results)
print("R score: ",r); print("Slope: ",slope); print("Intercept: ",intercept)

# reg = LinearRegression().fit(
# 	np.array(surrogate_results_list).reshape(-1,1),\
# 	np.array(gomlp_results).reshape(-1,1))
# print("Linear regression R^2: ",\
# 	reg.score(np.array(surrogate_results_list).reshape(-1,1),\
# 	np.array(gomlp_results).reshape(-1,1)))
# print("Linear regression coefficient: ", reg.coef_)

# plot scatter
plt.scatter(surrogate_results_list,gomlp_results,color="r",s=10)
plt.xlabel("Surrogate: MST Tree Intersection")
plt.ylabel("GOMLP")
plt.plot(np.linspace(min(surrogate_results_list),max(surrogate_results_list),1000),
	slope*np.linspace(min(surrogate_results_list),max(surrogate_results_list),1000)+intercept,label="R score: "+str(r)[:5])
plt.legend()
plt.show()

