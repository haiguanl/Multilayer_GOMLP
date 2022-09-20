import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import csv 

# design_file = "pins_BeagleBone_RevC_human.csv"
design_file = "pins_BeagleBone_RevC_human_singleLayer.csv"

design = np.loadtxt(design_file,delimiter=",",dtype=str)
header = [design[0,:]]
# print("Design header: ", header)

design_pd = pd.read_csv(design_file)
print("Design pd header: ",design_pd.head)
print("Total number of pins: ", design_pd.shape[0])

all_nets_list = list(set(design_pd["Net"].tolist()))
# print("All nets list: ",all_nets_list)
print("Total number of nets: ",len(all_nets_list))


# Visualize problem sets:
# net = all_nets_list[12]
# net = "VDD_1V8"
# print(design_pd[design_pd["Net"]==net]["X"].tolist())
# print(design_pd[design_pd["Net"]==net]["Y"].tolist())
net_number = 51
net_offset = 0
empty_list_count = 0 
non_empty_net = []
for i in range(net_offset,net_number+net_offset):
	net = all_nets_list[i]
	if not design_pd[design_pd["Net"]==net]["X"].tolist():
		print("Empty Net encountered...")
		empty_list_count += 1 
	else:
		non_empty_net.append(net)
		# plt.scatter(design_pd[design_pd["Net"]==net]["X"].tolist(),design_pd[design_pd["Net"]==net]["Y"].tolist(),s=8)

# for pin in range(design_pd[design_pd["Net"]==net].shape[0]):
	# print(design_pd[design_pd["Net"]==net][pin,:])
	# plt.scatter( design_pd[design_pd["Net"]==net])
print("Empty Net: ",empty_list_count)
# plt.show()

# Generate problem csv files
number_of_nets = 50 
number_of_nets_chosen = 20 
number_of_problems = 50 

net_indicator_list = [1 for i in range(number_of_nets_chosen)] + \
	[0 for i in range(number_of_nets-number_of_nets_chosen)]

print("Net to dump: ",non_empty_net)
# print("Net to dump length: ",len(non_empty_net))

with open("output_50nets.csv",'w', encoding='UTF8') as f:
	writer = csv.writer(f)
	writer.writerow(non_empty_net)
	# writer.writerow([1 for i in range(number_of_nets)])
	for i in range(number_of_problems):
		writer.writerow(np.random.permutation(net_indicator_list))




