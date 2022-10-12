#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[3]:



# -*- coding: utf-8 -*-
"""MLPGO_v1.1.ipynb

Automatically generated by Colaboratodry.

Original file is located at
    https://colab.research.google.com/drive/1ikLyXC1kem8mEyk0xSAt5j-kd3prXI_c

**files needed to run the module**
- pins_BeagleBone_RevC_human.csv

# imports
- scikit image library installation
"""

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

matplotlib_axes_logger.setLevel('ERROR')
# %matplotlib inline
import collections
import random
import pickle
import math
import pandas as pd
from os.path import exists as file_exists
from pathlib import Path
import itertools
import os
import csv 

pd.options.mode.chained_assignment = None
"""# HelperUtil 
- rendering metaballs
- handling colors and net types

"""


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
    def make_rgb_transparent(rgb, bg_rgb, alpha):
        return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]

    @staticmethod
    def display_metaballs(metaballs, fixed_metaballs, map, nets, net_index, alpha=1.0):
        for meta in metaballs:
            plt.clf()
            x, y, z = [], [], [] + net_index
            for handle in meta:
                x.append(handle.x * 1.0 / Parameter.scale)
                y.append(handle.y * 1.0 / Parameter.scale)

            mx, my, mz = list(x), list(y), list(z)
            fx, fy, fz = [], [], []
            for handle in fixed_metaballs:
                fx.append(handle.x * 1.0 / Parameter.scale)
                fy.append(handle.y * 1.0 / Parameter.scale)
                fz.append(handle.netid)
                x.append(handle.x * 1.0 / Parameter.scale)
                y.append(handle.y * 1.0 / Parameter.scale)
                z.append(handle.netid)

            if len(metaballs) > 0:
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

                clf = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(50, 50), max_iter=1000000,
                                    verbose=False, shuffle=True)
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
                for i, net in enumerate(nets):
                    color_sel = map[net]['rgb']
                    if alpha != 1.0:
                        color_sel_hsv = list(colorsys.rgb_to_hsv(color_sel[0], color_sel[1], color_sel[2]))
                        color_sel_hsv[2] = 1
                        color_sel = list(colorsys.hsv_to_rgb(color_sel_hsv[0], color_sel_hsv[1], color_sel_hsv[2]))
                    if alpha < 1:
                        alpha_col = Helper.make_rgb_transparent(color_sel, [1, 1, 1], 0.5)
                        plt.scatter(
                            [predict_mesh[j, 0] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                            [predict_mesh[j, 1] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                            c=[alpha_col])
                    else:
                        plt.scatter(
                            [predict_mesh[j, 0] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                            [predict_mesh[j, 1] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                            c=[color_sel], alpha=alpha)

            for i, net in enumerate(nets):
                color_sel = map[net]['rgb']
                plt.scatter([mx[j] for j in range(len(mz)) if mz[j] == i],
                            [my[j] for j in range(len(mz)) if mz[j] == i], facecolor='none', edgecolors=[color_sel])

            for i, net in enumerate(nets):
                for x, y in [(fx[j], fy[j]) for j in range(len(fz)) if fz[j] == i]:
                    plt.annotate(chr(65 + i) + ' ', xy=(x, y))
            plt.axis('off')
            plt.savefig('first.png')
            plt.show()
            plt.close()

    @staticmethod
    def display_fixed_pins(fixed_metaballs):
        fx, fy = [], []
        for handle in fixed_metaballs:
            fx.append(handle.x * 1.0 / Parameter.scale)
            fy.append(handle.y * 1.0 / Parameter.scale)
        x_min, x_max = 0, Parameter.board_width * 1.0 / Parameter.scale
        y_min, y_max = 0, Parameter.board_height * 1.0 / Parameter.scale
        h = 300
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
        plt.scatter(xx,yy,c = 'white')
        plt.scatter(fx, fy)
        plt.axis('off')
        plt.savefig('heatmap.png')
        plt.clf()
        image = cv2.imread('heatmap.png')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image


"""automatic Color generation code"""


class ColorHelper(object):
    @staticmethod
    def getHSVRangeFromRGB(rgb):
        cmax = max(rgb)
        cmin = min(rgb)
        delta = cmax - cmin
        h = 0
        if delta == 0:
            h = 0
        elif cmax == rgb[0]:
            h = 60 * (((rgb[1] - rgb[2]) / delta) % 6)
        elif cmax == rgb[1]:
            h = 60 * (((rgb[2] - rgb[0]) / delta) + 2)
        elif cmax == rgb[2]:
            h = 60 * (((rgb[0] - rgb[1]) / delta) + 4)
        h = h / 2
        s = 0
        if cmax != 0:
            s = delta / cmax
        s *= 255
        v = cmax
        v *= 255
        range = 5
        lower = np.array([h - range, s - range, v - range])
        upper = np.array([h + range, s + range, v + range])
        return lower, upper

    @staticmethod
    def generateNcolors(nets):
        error = 0
        count_nets = len(nets)
        res = {}
        # generation -------------------------
        color = [[np.random.uniform(0.0, 1.0) for _ in range(3)] for _ in range(count_nets)]
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        e = []
        for i in range(count_nets):
            cur = Ellipse(xy=(2 + 3 * i, 2), width=3, height=2, angle=50, facecolor=color[i])
            e.append(cur)
            ax.add_artist(cur)
            cur.set_clip_box(ax.bbox)
        ax.set_axis_off()
        ax.set_xlim(0, 4 * count_nets)
        ax.set_ylim(0, 4 * count_nets)
        plt.savefig("test.png", bbox_inches='tight')
        # return
        # plt.show()

        # detection ---------------------------
        image = cv2.imread('test.png')
        fit = 0
        notunique = False
        for cnt, col in enumerate(color):
            i2 = cv2.imread('test.png')
            lower_value, upper_value = ColorHelper.getHSVRangeFromRGB(col)
            res[nets[cnt]] = {'rgb': col, 'lower': lower_value, 'higher': upper_value}
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_value, upper_value)
            _, thresh = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) != 1:
                notunique = True
                break
            fit += len(contours)
        if notunique:
            return ColorHelper.generateNcolors(nets)
        return res


"""# Genetic Algorithm & Handle

**Handle :**

1. (x,y) coordinate
2. netid for fixed pins only

**Genetic Algorithm Parameters:**
1. number of generation
2. solution per generation, elite per generation
3. cross over percentage
4. noise ratio
"""


#  object structure for Handle
# (x,y) co-ordinate, angle for ellipse, height/width of vertical and horizontal chord
class Handle(object):
    def __init__(self, x, y, netid=None):
        self.x = int(x)
        self.y = int(y)
        self.netid = netid


# Genetic Algorithm implementation with cross over and mutation
class Genetic(object):
    # input: 
    # metaballs: list - moving handle 
    # nets: list - list of nets from csv input
    # fixed_metaballs: list - fixed handle generated from pins of net
    # parameters:
    # solution per population : to 
    def __init__(self, nets, fixed_metaballs, color_map, num_handles, image_dtm,results_file,problem_id):
        # GA parameters 
        self.num_of_generations = 25
        self.sol_per_pop = 10
        self.nets = nets
        self.fixed_metaballs = fixed_metaballs
        self.color_map = color_map
        self.num_handles = num_handles
        self.image_dtm = image_dtm
        self.net_index = []
        # default 
        # self.weights = {'num_islands': 0.9, 'disjoint_distance': 0.9, 'island_length': 0.30 , 'contour_dtm':0.50}
        self.weights = {'num_islands': 0.9, 'disjoint_distance': 0.9, 'island_length': 0.00 , 'contour_dtm':0.00}
        self.best_fitness = []
        self.results_file = results_file
        self.problem_id = problem_id

    def init_population(self):
        new_population = []
        net_to_pt = collections.defaultdict(list)
        for meta in self.fixed_metaballs:
            net_to_pt[meta.netid].append(meta)

        # moving points distribution balancing among nets with unequal fix pins
        max_pts = max([len(net_to_pt[i]) for i in range(len(self.nets))])
        distribution = [max(0, max_pts - len(net_to_pt[i])) for i in range(len(self.nets))]
        per_net = int((self.num_handles - sum(distribution)) / len(self.nets))
        for i in range(len(distribution)):
            distribution[i] += per_net
            #edf
       
        self.num_handles = sum(distribution)
        if self.num_handles == 0:
          distribution[0] = 1
          self.num_handles = 1
        print(self.num_handles, distribution )
        # create N population with fixed and moving points
        for i in range(self.sol_per_pop):
            cur = []
            # in each solution distribute points fairly 
            for net_id in range(len(distribution)):
                # make n points for net_id net
                for mid in range(distribution[net_id] + 1):
                    sel_pt = random.randint(0, len(net_to_pt[net_id]) - 1)
                    sel_pt = net_to_pt[net_id][sel_pt]
                    max_x, min_x = min(-1 + (Parameter.board_width), 1.25 * sel_pt.x), 0.75 * sel_pt.x
                    max_y, min_y = min(-1 + (Parameter.board_height), 1.25 * sel_pt.y), 0.75 * sel_pt.y
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(min_y, max_y)
                    cur.append(deepcopy(Handle(x, y)))
            new_population.append(cur)

        temp = [] + distribution
        for i in range(len(temp)):
            self.net_index += [i] * (temp[i] + 1)
        return new_population

    def calc_fitness(self, meta, index,gen):
        x, y, z = [], [], [] + self.net_index
        for handle in meta:
            x.append(handle.x * 1.0 / Parameter.scale)
            y.append(handle.y * 1.0 / Parameter.scale)
        fx, fy, fz = [], [], []
        for handle in self.fixed_metaballs:
            fx.append(handle.x * 1.0 / Parameter.scale)
            fy.append(handle.y * 1.0 / Parameter.scale)
            fz.append(handle.netid)
            x.append(handle.x * 1.0 / Parameter.scale)
            y.append(handle.y * 1.0 / Parameter.scale)
            z.append(handle.netid)

        # feature engineering

        x = np.array(x)
        y = np.array(y)
        fx = np.array(fx)
        fy = np.array(fy)

        data = np.c_[x, y, np.multiply(x, y)]
        for i in range(2, Parameter.order + 1):
            data = np.c_[
                data, (np.sin(0.01 * i * np.pi * x)).T, (np.cos(0.01 * i * np.pi * x)).T, 0.01 * np.power(x, i)]
            data = np.c_[
                data, (np.sin(0.01 * i * np.pi * y)).T, (np.cos(0.01 * i * np.pi * y)).T, 0.01 * np.power(y, i)]

        test_data = np.c_[fx, fy, np.multiply(fx, fy)]
        for i in range(2, Parameter.order + 1):
            test_data = np.c_[
                test_data, (np.sin(0.01 * i * np.pi * fx)).T, (np.cos(0.01 * i * np.pi * fx)).T, 0.01 * np.power(fx, i)]
            test_data = np.c_[
                test_data, (np.sin(0.01 * i * np.pi * fy)).T, (np.cos(0.01 * i * np.pi * fy)).T, 0.01 * np.power(fy, i)]

        clf = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(50, 50), max_iter=1000000,
                            verbose=False, shuffle=True)
        clf.fit(data, z)
        score = clf.score(test_data, fz)
        # cost scores -1 is default in case score is less than 1.0
        num_islands = -1
        disjoint_distance = -1
        island_length = -1
        contours_dtm = -1
        # if score == 1.0:
        # print("Debug score: ",score)
        if score > 0.5:

            num_islands = 0
            disjoint_distance = 0
            island_length = 0
            contours_dtm = 0
            plt.clf()
            h = 300
            x_min, x_max = 0, Parameter.board_width * 1.0 / Parameter.scale
            y_min, y_max = 0, Parameter.board_height * 1.0 / Parameter.scale
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))

            # feature engineering
            grid_data = np.c_[xx.ravel(), yy.ravel(), np.multiply(xx.ravel(), yy.ravel())]
            for i in range(2, Parameter.order + 1):
                grid_data = np.c_[grid_data, (np.sin(0.01 * i * np.pi * xx.ravel())).T, (
                    np.cos(0.01 * i * np.pi * xx.ravel())).T, 0.01 * np.power(xx.ravel(), i)]
                grid_data = np.c_[grid_data, (np.sin(0.01 * i * np.pi * yy.ravel())).T, (
                    np.cos(0.01 * i * np.pi * yy.ravel())).T, 0.01 * np.power(yy.ravel(), i)]

            zz = clf.predict(grid_data)
            predict_mesh = np.transpose(np.vstack((xx.ravel(), yy.ravel(), zz)))
            for i, net in enumerate(self.nets):
                plt.scatter([predict_mesh[j, 0] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                            [predict_mesh[j, 1] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                            c=[self.color_map[net]['rgb']])

            for i, net in enumerate(self.nets):
                plt.scatter([fx[j] for j in range(len(fz)) if fz[j] == i],
                            [fy[j] for j in range(len(fz)) if fz[j] == i], c=[self.color_map[net]['rgb']])

            plt.axis('off')
            plt.savefig("tmp/"+"problem"+str(self.problem_id)+"gen"+str(gen)+"ind"+str(index) + '.png')
            # fig.show()

            image = cv2.imread("tmp/"+"problem"+str(self.problem_id)+"gen"+str(gen)+"ind"+str(index) + '.png')
            # weights = {'num_islands':100,'disjoint_distance':1,'island_length': 1,'DMT':1}

            for net in self.nets:
                lower_value, upper_value = self.color_map[net]['lower'], self.color_map[net]['higher']
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_value, upper_value)
                _, thresh = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                num_islands += len(contours)
                for c in contours:
                    island_length += cv2.arcLength(c, True)
                    for x,y in c[:,0,:]:
                        x = min(x, self.image_dtm.shape[0]-1)
                        y = min(y, self.image_dtm.shape[1] - 1)
                        contours_dtm += self.image_dtm[x,y]
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
            if num_islands == len(self.nets):
                # print('converged')
                # 07/07/22-debug
                # pickle.dump(clf, open('converged_model.sav', 'wb'))
                plt.clf()
                plt.close()
                for i, net in enumerate(self.nets):
                    color_sel = self.color_map[net]['rgb']
                    color_sel_hsv = list(colorsys.rgb_to_hsv(color_sel[0], color_sel[1], color_sel[2]))
                    color_sel_hsv[2] = 1
                    color_sel = list(colorsys.hsv_to_rgb(color_sel_hsv[0], color_sel_hsv[1], color_sel_hsv[2]))
                    color_sel = Helper.make_rgb_transparent(color_sel, [1, 1, 1], 0.5)
                    plt.scatter(
                        [predict_mesh[j, 0] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                        [predict_mesh[j, 1] for j in range(predict_mesh.shape[0]) if int(predict_mesh[j, 2]) == i],
                        c=[color_sel], alpha=0.5)
                for i, net in enumerate(self.nets):
                    for x, y in [(fx[j], fy[j]) for j in range(len(fz)) if fz[j] == i]:
                        plt.annotate(chr(65 + i) + ' ', xy=(x, y))
                        # plt.show()
                plt.axis('off')
                if Parameter.SAVE_FLAG:
                  Path(Parameter.SAVE_LOCATION+"/problem"+str(self.problem_id)+"/"+str(index)).mkdir(parents=True, exist_ok=True)
                  plt.savefig(Parameter.SAVE_LOCATION+"/problem"+str(self.problem_id)+"/"+str(index)+'/'+str(gen)+'.png')
            plt.clf()
            plt.close()

        fit = [num_islands, island_length, disjoint_distance, contours_dtm,index]
        return fit

    def calc_pop_fitness(self, population,gen):
        plt.clf()
        fitness_data = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            index = [i for i in range(len(population))]
            gen = [gen for _ in range(len(population))]
            fitness = executor.map(self.calc_fitness, population, index,gen)
            for fit_val in fitness:
                fitness_data.append(fit_val)

        # island.sort(key=lambda x: x[0])
        col_names = ['num_islands','island_length','disjoint_distance','contour_dtm']
        fitness_data = pd.DataFrame(fitness_data, columns =col_names+['index'])
        fitness_data_copy = fitness_data.copy()
        for col_name in col_names:
            fitness_data.loc[:, col_name][fitness_data.loc[:,col_name] == -1] = max(fitness_data.loc[:,col_name]) + 1
            _max = max(fitness_data.loc[:,col_name])
            fitness_data.loc[:, col_name] /= (_max+0.1)
        fitness_data.loc[:, 'contour_dtm'] *= -1

        # {'num_islands': 0.25, 'disjoint_distance': 0.25, 'island_length': 0.25, 'contour_dtm': 0.25}
        fitness = []
        for i in range(fitness_data.shape[0]):
            fit = 0
            for col_name in col_names:
                fit += self.weights[col_name] * fitness_data.loc[i,col_name]
            fitness.append([fit,population[fitness_data.loc[i,'index']],list(fitness_data_copy.iloc[i,:])])
        fitness.sort(key=lambda x: x[0])
        return np.array(fitness, dtype=object)

    def gen_single(self, wheel):
        pop_index1 = np.random.randint(0, len(wheel))
        cur = deepcopy(wheel[pop_index1][1])
        org_idx = cur[0]
        pop_index2 = None
        # avoiding the duplicate population selection by wheel selection
        while pop_index2 == None or pop_index2[0] == org_idx:
            pop_index2 = np.random.randint(0, len(wheel))
            pop_index2 = wheel[pop_index2]

        second = pop_index2[1]
        # cross-over
        for __ in range(len(cur)):
            if random.random() <= 0.30:
                handle_id = np.random.randint(0, len(cur) - 1)
                cur[handle_id] = deepcopy(second[handle_id])

        # mutation
        for handle in range(self.num_handles):
            if random.random() <= .5:
                cur[handle].x += random.gauss(0, Parameter.mutation['sd_x'])
                if cur[handle].x < 0:
                  cur[handle].x = 0
                if cur[handle].x >= Parameter.board_width:
                  cur[handle].x = Parameter.board_width - 1
                cur[handle].y += random.gauss(0, Parameter.mutation['sd_y'])
                if cur[handle].y < 0:
                  cur[handle].y = 0
                if cur[handle].y >= Parameter.board_height:
                  cur[handle].y = Parameter.board_height - 1
        return cur

    def gen_new_population(self, elite, wheel):
        final_pop = list(elite)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            population = [executor.submit(self.gen_single, wheel) for _ in range(2 * self.sol_per_pop)]
            for individual in concurrent.futures.as_completed(population):
                final_pop.append(individual.result())
        return final_pop

    def run(self, verbose=True):
        best_fit = float('inf')
        best_params = []
        new_population = self.init_population()
        gen_rate = []
        w_len = []
        for gen in range(self.num_of_generations):
            fitness = self.calc_pop_fitness(new_population,gen)
            sorted_fitness = []
            fit = []
            if verbose: print('\nRunning Generation: ', gen, "->", [fitness[i][0] for i in range(len(fitness))])
            for i in range(self.sol_per_pop):
                sorted_fitness.append(fitness[i][1])
                fit.append(fitness[i][0])

            new_population = sorted_fitness
            # adaptive weight logic for island length
            if gen >3:
              data = np.array(self.best_fitness)
              cnt = 0
              for num in data[:,0]:
                if num == len(self.nets):
                  cnt += 1
              gen_rate.append(cnt/len(data[:,0]))
              if len(gen_rate) > 2 and (gen_rate[-1] == 1.0 or ((gen_rate[-2] - gen_rate[-1])/(gen_rate[-2]+.01))>0.0001):
                self.weights['island_length'] = min(10**3,self.weights['island_length']*10)
              w_len.append(self.weights['island_length'])

            
            if len(self.best_fitness) > 0:
              self.best_fitness.append(fitness[0][2])
            else:
              self.best_fitness = [fitness[0][2]]
            # Note: plot loss term
            if gen !=0 and gen % 1 == 10 and Parameter.PLOT_FLAG:
              data = np.array(self.best_fitness)
              plt.plot(np.arange(len(self.best_fitness)), data[:, 0],label = 'island')
              plt.legend()
              plt.show()
              plt.plot(np.arange(len(self.best_fitness)), data[:, 1],label = 'island length')
              plt.legend()
              plt.show()
              plt.plot(np.arange(len(self.best_fitness)), data[:,2],label = 'disjoint length')
              plt.legend()
              plt.show()
              plt.plot(np.arange(len(self.best_fitness)), data[:,3],label = 'DTM')
              plt.legend()                                                                                                                                                                                                                                                                                                                                                                    
              plt.show()
              plt.plot(np.arange(len(gen_rate)), gen_rate,label = 'convergance_rate')
              plt.legend()                                                                                                                                                                                                                                                                                                                                                                    
              plt.show()
              plt.plot(np.arange(len(w_len)), w_len,label = 'weight of island length')
              plt.legend()                                                                                                                                                                                                                                                                                                                                                                    
              plt.show()

            if best_fit > fitness[0][0]:
              best_fit = fitness[0][0]
              best_params = fitness[0][2]
            fitness = fit
            if verbose: print('Running Generation: ', gen, "->", fitness)
            # 07/07/22-debug
            # if gen % 5 == 0:
                # Helper.display_metaballs([new_population[0]], self.fixed_metaballs, self.color_map, self.nets,
                                         # self.net_index, 0.5)
            if len(self.nets) in fitness:  # convergance criteria
                # Helper.display_metaballs([new_population[0]], self.fixed_metaballs, self.color_map, self.nets,
                                         # self.net_index, 0.5)
                return best_params

            elite = []
            wheel = []
            _max = max(fitness)

            for e in range(self.sol_per_pop):
                elite.append(new_population[e])
                wheel.append([e, new_population[e]])
                repeat = int(_max - fitness[e])
                for _ in range(repeat):
                    wheel.append([e, new_population[e]])
            new_population = self.gen_new_population(elite, wheel)
        print('\nFinal result',np.array(self.best_fitness)[:, 0])
        with open(self.results_file,"a") as f: 
            writer = csv.writer(f)
            writer.writerow(str(min(np.array(self.best_fitness)[:, 0]))+"\n")
        # Helper.display_metaballs([new_population[0]], self.fixed_metaballs, self.color_map, self.nets, self.net_index,
                                 # 0.5)
        return best_params


"""#**Runner**"""

class Parameter(object):
    board_height, board_width = 6 * 1e10, 9 * 1e10
    order = 3
    scale = 1e10
    mutation = {'sd_x': board_width * .1, 'sd_y': board_height * .1}
    #save best generation png in generation folders
    SAVE_FLAG = True
    # SAVE_LOCATION = '/content/drive/MyDrive/converged/'
    SAVE_LOCATION = './converged/'

    #plot progress graph for cost function
    PLOT_FLAG = True

class GOMLP2:

  @staticmethod
  def run(pin_csv_file, color_dict, layer_name, nets,results_file,problem_id):
      pins = pd.read_csv(pin_csv_file)
      if not file_exists(color_dict):
          color_map = ColorHelper.generateNcolors(nets)
          with open(color_dict, 'wb') as file:
              pickle.dump(color_map, file)
      else:
          with open(color_dict, 'rb') as file:
              color_map = pickle.load(file)

      fixed_metaballs = []
      # fix handle creation
      for i, layer_name in enumerate(layer_name):
          layer = pins.loc[pins['Layer'] == layer_name]
          for j in range(len(nets)):
              net_pins = layer.loc[layer['Net'] == nets[j]]
              net_pins = net_pins[['X', 'Y']]
              net_coord = net_pins.to_numpy()
              fixed_metaballs += Helper.convert_to_handle_list(net_coord, j)

      num_handle = 30
      # print(color_map)
      image = Helper.display_fixed_pins(fixed_metaballs)
      image[np.where(image < 255)] = 0
      image[np.where(image == 255)] = 1
      image_dtm = scipy.ndimage.distance_transform_edt(image)
      # 07/07/22-debug
      # plt.imshow(image_dtm)
      # plt.colorbar()
      # plt.savefig('heatmap.png')
      # plt.clf()
      # Helper.display_metaballs([[]],fixed_metaballs,color_map,nets,[])

      genAlgo = Genetic(nets, fixed_metaballs, color_map, num_handle,image_dtm,results_file,problem_id)
      best_params = genAlgo.run(True)
      return best_params





# In[4]:


if __name__ == '__main__':

    # Main API
    pin_csv_file = "pins_BeagleBone_RevC_human_singleLayer.csv"
    # pin_csv_file = "pins_BeagleBone_RevC_human.csv"

    color_dict = 'color_map.pkl'
    layer_name = ["LYR5_PWR"]

    problem_files = "output_50nets.csv"
    problems = np.loadtxt(problem_files,delimiter=",",dtype=str)
    netlist = problems[0,:]
    results_file = "gomlp_results_50nets.csv"
    if not os.path.exists(results_file):
        os.system("touch "+results_file)

    print("netlist: ",problems[0,:])

    # for problem_ind in range(1,51):
    
    problem_ind = 1
    # Previous 
    internal_nets_list = [netlist[j] for j,i in enumerate(problems[problem_ind,:]) if i=="1"]
    # With hierarchical clustering results 
    output_file = "layer_assignment_emd.pkl"
    with open("results/"+output_file, 'rb') as f:
        cluster_dict = pickle.load(f)
    print("cluster_dict: ",cluster_dict)

    for key,val in cluster_dict.items():
        internal_nets_list = val
        problem_ind = key
        print("internal_nets_list: ",internal_nets_list)
        print("net_list length: ",len(internal_nets_list))    
        # short_net_name = list(itertools.permutations("ABCDE"))
        nets = []
        for ind,net in enumerate(internal_nets_list):
            # print("short_net_name[ind]: ","".join(short_net_name[ind]))
            # nets.append("".join(short_net_name[ind]))
            nets.append(net)

        print("Nets: ",nets)
        GOMLP2.run(pin_csv_file, color_dict, layer_name, nets,results_file,problem_ind)


# In[ ]:




