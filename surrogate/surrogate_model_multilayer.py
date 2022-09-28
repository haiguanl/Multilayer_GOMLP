# Commented out IPython magic to ensure Python compatibility.
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
from copy import deepcopy
import cv2
import concurrent.futures
import functools
import colorsys
import matplotlib.image as mpimg
import scipy.interpolate
import matplotlib.colors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString

import itertools

matplotlib_axes_logger.setLevel('ERROR')
# %matplotlib inline
import collections
import random
import pickle
import math
import pandas as pd
from os.path import exists as file_exists
from pathlib import Path
# from google.colab.patches import cv2_imshow

pd.options.mode.chained_assignment = None



#  object structure for Handle
# (x,y) co-ordinate, angle for ellipse, height/width of vertical and horizontal chord
class Handle(object):
	def __init__(self, x, y, netid=None):
		self.x = int(x)
		self.y = int(y)
		self.netid = netid


class Helper(object):
	# convert pins into list of handles
	# output: list of handles
	@staticmethod
	def convert_to_handle_list(arr, net_id):
		cur = []
		for pins in arr:
			cur.append(Handle(pins[0], pins[1], net_id))
		return cur

	@staticmethod
	def calculate_cost_from_image(image,color_map,nets):
		num_islands = 0
		disjoint_distance = -1
		island_length = -1
		for net in nets:
			lower_value, upper_value = color_map[net]['lower'], color_map[net]['higher']
			hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, lower_value, upper_value)
			_, thresh = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
			contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			num_islands += len(contours)
			for c in contours:
				island_length += cv2.arcLength(c, True)
			if len(contours) != 1:
				c_pts = []
				for c in contours:
					M = cv2.moments(c)
					cX = int(M["m10"] / (M["m00"] + 0.1))
					cY = int(M["m01"] / (M["m00"] + 0.1))
					c_pts.append((cX, cY))
				min_dst = []
				for i in range(len(c_pts) - 1):
					_min = float('inf')
					for j in range(i + 1, len(c_pts)):
						_min = min(_min, math.hypot(c_pts[i][0] - c_pts[j][0], c_pts[i][1] - c_pts[j][1]))
					min_dst.append(_min)
				disjoint_distance += sum(min_dst)
		num_islands -= len(nets)
		return [num_islands, disjoint_distance, island_length]

	@staticmethod 
	def generate_tree_for_problems(temp_net_coord_total):
		# Generate minimum spanning tree for all nets and check intersection 
		net_line_segment_total = []
		for net in temp_net_coord_total[0]:
			net_line_segment = []
			# print(net)
			# print("Debug: new net\n")
			plt.scatter(net[:,0],net[:,1],s=10)
			# generate MST
			dis_mat = np.zeros((net.shape[0],net.shape[0]))
			for i in range(net.shape[0]-1):
				for j in range(i+1,net.shape[0]):
					dis_mat[i,j] = np.linalg.norm(
						net[i,[0,1]]-net[j,[0,1]])
			X = csr_matrix(dis_mat)
			Tcsr = minimum_spanning_tree(X); T = Tcsr.toarray().astype(int)
			indices_MST = np.nonzero(T)
			# print("Debug indices_MST: ",indices_MST)
			for i in range(len(indices_MST[0])):
				pin1 = net[indices_MST[0][i],:]
				pin2 = net[indices_MST[1][i],:]
				plt.plot([pin1[0],pin2[0]],[pin1[1],pin2[1]],'k')
				net_line_segment.append([(pin1[0],pin1[1]),(pin2[0],pin2[1])])
			net_line_segment_total.append(net_line_segment)

			# Tcsr = mst(X)
		# print("Debug total line segment: ",len(net_line_segment_total))
		# count the number of intersections of all nets 
		intersections = 0 
		for i,j in enumerate(net_line_segment_total):
			for k in range(i+1,len(net_line_segment_total)):
				net1 = net_line_segment_total[i]
				net2 = net_line_segment_total[k]
				# print("Net1: ",net1)
				# print("Net2: ",net2)
				for m in net1:
					for n in net2: 
						# print("Debug: n",n)
						# print("Debug: m",m)
						line1 = LineString([m[0],m[1]])
						line2 = LineString([n[0],n[1]])
						if line1.intersects(line2): 
							intersections+=1

		# for i in range(len(net_line_segment_total)-1):
			# for j in range(i+1,len(net_line_segment_total)):
				# line1 = LineString([net_line_segment_total[i][0],net_line_segment_total[i][1]])
				# line2 = LineString([net_line_segment_total[j][0],net_line_segment_total[j][1]])
				# if line1.intersect(line2):
					# intersections += 1 
					# print("Check ...")
		# print("Debug intersections: ", intersections)		
		# plt.show()
		# plt.show(block=False)
		# plt.pause(2)
		# plt.close()
		return intersections


class Parameter(object):
	board_height, board_width = 6 * 1e10, 9 * 1e10
	order = 3
	scale = 1e10


class SurrogateModel(object):
	def __init__(self,color_map):
		self.color_map = color_map
		# print("self.color_map: ",self.color_map)

	def train_and_generate_boundary(self,model,handles,index,nets,verbose = True):
		plt.clf()
		self.nets = nets
		x, y, z = [], [], []
		for handle in handles:
			x.append(handle.x * 1.0 / Parameter.scale)
			y.append(handle.y * 1.0 / Parameter.scale)
			z.append(handle.netid)

		if len(x) > 0:
			x_min, x_max = 0, Parameter.board_width * 1.0 / Parameter.scale
			y_min, y_max = 0, Parameter.board_height * 1.0 / Parameter.scale

			x = np.array(x)
			y = np.array(y)

			data = np.c_[x, y, np.multiply(x, y)]

			for i in range(2, Parameter.order + 1):
				data = np.c_[
				data, (np.sin(0.01 * i * np.pi * x)).T, (np.cos(0.01 * i * np.pi * x)).T, 0.01 * np.power(x, i)]
				data = np.c_[
				data, (np.sin(0.01 * i * np.pi * y)).T, (np.cos(0.01 * i * np.pi * y)).T, 0.01 * np.power(y, i)]

			clf = model
			clf.fit(data, z)
			h = 300
			xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
			grid_data = np.c_[xx.ravel(), yy.ravel(),
			np.multiply(xx.ravel(), yy.ravel())]

			for i in range(2, Parameter.order + 1):
				grid_data = np.c_[grid_data, (np.sin(0.01 * i * np.pi * xx.ravel())).T, (
				np.cos(0.01 * i * np.pi * xx.ravel())).T, 0.01 * np.power(xx.ravel(), i)]
				grid_data = np.c_[grid_data, (np.sin(0.01 * i * np.pi * yy.ravel())).T, (
				np.cos(0.01 * i * np.pi * yy.ravel())).T, 0.01 * np.power(yy.ravel(), i)]
			zz = clf.predict(grid_data)

			predict_mesh = np.transpose(np.vstack((xx.ravel(), yy.ravel(), zz)))
			for i, net in enumerate(self.nets):
				plt.scatter(
				[predict_mesh[j, 0] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
				[predict_mesh[j, 1] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
				c=[self.color_map[net]['rgb']])

		for i, net in enumerate(self.nets):
			plt.scatter([x[j] for j in range(len(z)) if z[j] == i],
			[y[j] for j in range(len(z)) if z[j] == i], c=[self.color_map[net]['rgb']])
		plt.axis('off')
		plt.savefig("tmp/"+str(index)+'temp.png')
		plt.close()
		image = cv2.imread("tmp/"+str(index)+'temp.png')
		#     if verbose:
		#       cv2_imshow(image)
		return image

class SurrogateRunner(object):
	@staticmethod
	def run(color_dict,pin_csv_file,layer_name,nets,model,index,tree_model):
		# print(color_dict, pin_csv_file, layer_name,nets)
		color_map = None
		with open(color_dict, 'rb') as file:
			color_map = pickle.load(file)

		pins = pd.read_csv(pin_csv_file)

		handles = []; temp_net_coord_total = []
		for i, layer_name in enumerate(layer_name):
			layer = pins.loc[pins['Layer'] == layer_name]
			# print("Debug layer: ",layer)
			temp_net_coord = []
			for j in range(len(nets)):
				net_pins = layer.loc[layer['Net'] == nets[j]]
				net_pins = net_pins[['X', 'Y']]
				net_coord = net_pins.to_numpy()
				temp_net_coord.append(net_coord)
				# print("Debug Net coord: ",net_coord)
				handles += Helper.convert_to_handle_list(net_coord, j)
				# print("Debug handles: ",handles)
			temp_net_coord_total.append(temp_net_coord)
		# print("Debug temp_net_coord_total: ",temp_net_coord_total)
		if tree_model:
			intersections = Helper.generate_tree_for_problems(temp_net_coord_total)
			# cost_func = [index]
			# cost_func += intersections
			# surrogate_model = SurrogateModel(color_map)
			# return cost_func

		surrogate_model = SurrogateModel(color_map)
		boundary_image = surrogate_model.train_and_generate_boundary(model,handles,index,nets)
		cost_func = [index]
		if tree_model: 
			cost_func += [intersections]
		else:
			cost_func += Helper.calculate_cost_from_image(boundary_image,color_map,nets)
		# print("Debug cosf_func: ",cost_func)
		return cost_func



def group_test_runner(model,list_nets,tree_model):
	color_dict = 'color_map.pkl'
	pin_csv_file = "pins_BeagleBone_RevC_human_singleLayer.csv"
	layer_name = ["LYR5_PWR"]
	results = collections.defaultdict(list)
	ei_results = collections.defaultdict(int)
	# tree_model = True
	# print("Debug: list_nets: ",len(list_nets))
	with concurrent.futures.ProcessPoolExecutor() as executor:
		# print("Running in parallel...")
		index = [i for i in range(len(list_nets))]
		models = [model]*len(list_nets)
		tree_model = [tree_model]*len(list_nets)
		color_dict = [color_dict]*len(list_nets)
		pin_csv_file = [pin_csv_file]*len(list_nets)
		layer_name = [layer_name]*len(list_nets)
		cost_func = executor.map(SurrogateRunner.run, color_dict,pin_csv_file,layer_name,list_nets,models,index,tree_model)
		print("cost_func: ",cost_func)

		for cf in cost_func: 
			ei_results[cf[0]] = cf[1]
			results[cf[0]] = cf
	return ei_results   # ?? Debug: indent

# Could change to not concurrent


def generate_test_cases_from_csv(combination_csv,index):
	list_nets = []
	comb = pd.read_csv(combination_csv)
	print("Comibnations of nets: ",comb)
	column_names = list(comb)
	# print("column_names: ",column_names)
	# internal_nets = {'GNDA_ADC':'A','VDDS_DDR':'B','VDD_PHYA':'C','VDD_CORE':'D',
	#                'VDD_1V8FT':'E','GND_EARTH':'F','VDD_MPU':'G','VDD_3V3B':'H',
	#                'VDD_1V8':'I'}
	
	# internal_nets_list = [
	# 'AIN6', 'DDR_A1', 'P_MUXOUT', 'RMII1_CRS_DV', 'MMC0_CMD', 'AIN1', 
	# 'NINT/TXER/TXD4', 'DDR_D8', 'TXP', 'PDDR_D11', 'NC1', 'VDD_FTVPLL',
	#  'DDR_D14', 'GPIO2_23', 'OSC0_OUT', 'DDR_VREF', 'EMU4R', 'LDO3', 'LEDBA',
	#   'UART0_RX', 'NC2', 'PDDR_D4', 'DDR_A2', 'GPIO1_15', 'PDDR_A9', 'TDIS',
	#    'LEDCA', 'DDR_A12', 'OSC1_OUT1', 'PDDR_A4', 'UART4_TXD', 'PDDR_D7', 'LDO4',
	#     'CAP_VDD_SRAM_MPU', 'CAP_VBB_MPU', 'UART2_RXD', 'TIMER4', 'AIN7', 'TIMER5',
	#      'CAP_VDD_SRAM_CORE', 'LEDDC', 'USB1_DRVVBUS', 'BCBUS0', 'BL_SINK1', 'GPIO1_16',
	#       'JTAG_EMU0', 'COL/CRS_DV/MODE2', 'GRNA', 'O\\C\\S\\2\\', 'GPIO1_30'
	#  ]
	internal_nets_list = column_names
	short_net_name = list(itertools.permutations("ABCDE"))
	internal_nets = {}
	for ind,net in enumerate(internal_nets_list):
		# print("short_net_name[ind]: ","".join(short_net_name[ind]))
		# internal_nets[net] = "".join(short_net_name[ind])
		internal_nets[net] = net


	for idx in index:
		cur = []
		for cid,col in enumerate(comb.iloc[idx,:]):
			# print("comb: ",cid,col)
			if col==1:
				cur.append(internal_nets[column_names[cid]])
		list_nets.append(cur)
	return list_nets

def save_dict(results,name):
	results = pd.DataFrame(data=results, index=[0])
	results = (results.T)
	results.to_excel(name+'.xlsx')


def plot_results():
	gomlp = pd.read_excel('GOMLP.xlsx')
	mlp = pd.read_excel('MLP.xlsx')
	svc = pd.read_excel('SVC.xlsx')
	gomlp = gomlp.sort_values(by=0)

	plt.plot(np.arange(36),gomlp.iloc[:,1],label ='GOMLP')
	plt.plot(np.arange(36),mlp.iloc[gomlp.iloc[:,0],1],label = 'MLP')
	plt.title('MLP vs GOMLP all upto 6 nets experiments')
	plt.legend()
	plt.show()

	plt.plot(np.arange(36),gomlp.iloc[:,1],label ='GOMLP')
	plt.plot(np.arange(36),svc.iloc[gomlp.iloc[:,0],1],label = 'SVC')
	plt.title('SVC vs GOMLP all upto 6 nets experiments')
	plt.legend()
	plt.show()


if __name__ == "__main__":


	# Main API: 

	# Get test case:
	# testcase_ids = [i for i in range(129)]
	number_of_problems = 1
	testcase_ids = [i for i in range(number_of_problems)]
	# test_cases = generate_test_cases_from_csv('output_6plus.csv',testcase_ids) 
	test_cases = generate_test_cases_from_csv('output_50nets.csv',testcase_ids) 

	print("Test cases: ",test_cases)

	# Get combinations of nets (pair wise)
	net_combinations = []
	for i in range(len(test_cases[0])-1):
		for j in range(i+1,len(test_cases[0])):
			netA = test_cases[0][i]; netB = test_cases[0][j]
			net_combinations.append([netA,netB])
	print("net_combinations: ",net_combinations)

	# Run surrogate model for all test cases 
	run_surrogate = True; tree_model = False
	if run_surrogate:  
		for max_iteration in [1000]:
			model = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(50, 50), max_iter=max_iteration,
		    verbose=False, shuffle=True)
			# model = SVC(gamma='auto')
			results = group_test_runner(model,test_cases,tree_model)
			print("Results: ", results)
			# save_dict(results,'MLP_ite{ite}'.format(ite=max_iteration))
			save_dict(results,'Tree')


	# model = SVC(kernel='rbf')
	# results = group_test_runner(model,test_cases)
	# save_dict(results,'SVC')

	# plot_results()	

	# Small test case
	# color_dict = 'color_map.pkl'
	# pin_csv_file = "pins_BeagleBone_RevC_human.csv"
	# layer_name = ["LYR5_PWR"]
	# nets = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

	# model = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(50, 50), max_iter=1000000,
	#                               verbose=False, shuffle=True)

	# print(SurrogateRunner.run(color_dict,pin_csv_file,layer_name,nets,model,0))







