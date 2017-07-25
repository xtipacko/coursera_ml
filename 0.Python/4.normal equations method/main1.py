import warnings
warnings.filterwarnings("ignore")

import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from plot3d import Plot
from pprint import pprint
from copy import deepcopy

def filter_float_tupples(*args):
    try:
        result = []
        for arg in args:
            result.append(float(arg))
        return result
    except:
        return None

datapoints = []
with open('houseprices.csv', 'r') as datafile:
    csv_dict = csv.DictReader(datafile)
    for i,row in enumerate(csv_dict):
        datapoint = filter_float_tupples(row['sqft_living'], 
                                         row['yr_built'], 
                                         row['price'])
        if datapoint and not i % 50:
            datapoints.append(datapoint)

prepared_dps = []
for datapoint in datapoints:
    sqft_living = datapoint[0]
    age         = 2017 - datapoint[1] 
    price       = datapoint[2]
    prepared_dps.append([sqft_living, age, price])


np_datapoints = np.array(prepared_dps, dtype=float)

array_sqft_living = np_datapoints[:,0]
array_age         = np_datapoints[:,1]
array_price       = np_datapoints[:,2]

Xarray_sqft_living = deepcopy(array_sqft_living)
Xarray_age         = deepcopy(array_age)
Xarray_price       = deepcopy(array_price)


mean_sqft_living = np.mean(array_sqft_living)
mean_age         = np.mean(array_age)
mean_price       = np.mean(array_price)


std_sqft_living = np.std(array_sqft_living)
std_age         = np.std(array_age)
std_price       = np.std(array_price)


scaled_array_sqft_living = (array_sqft_living - mean_sqft_living) / std_sqft_living
scaled_array_age         = (array_age         - mean_age)         / std_age 
scaled_array_price       = (array_price       - mean_price)       / std_price

bias_vector = np.array([1]*scaled_array_sqft_living.size)

scaled_np_dps  = np.column_stack((bias_vector,
                                  scaled_array_sqft_living, 
                                  scaled_array_age, 
                                  scaled_array_price))



X  = np.column_stack((bias_vector,
                      scaled_array_sqft_living, 
                      scaled_array_age ))
y = scaled_array_price


plot = Plot(np.array([1,1,1])*5)

plot.ax.scatter(scaled_array_sqft_living,
                scaled_array_age,
                scaled_array_price,
                c='g', marker='.')




#C+Ax1+Bx2 = z


def grad(datapoints, C, A, B):
    m = datapoints.shape[0] # amount of rows in matrix
    parameter_vector = np.array([C,A,B,-1])
    hypotheses_minus_price_vector = datapoints.dot(parameter_vector)
    grad = np.transpose(datapoints[:,0:3]).dot(hypotheses_minus_price_vector) /m
    return grad

def SSE(datapoints, C, A, B):
    m = datapoints.shape[0] # amount of rows in matrix
    parameter_vector = np.array([C,A,B,-1])
    hypotheses_minus_price_vector = datapoints.dot(parameter_vector)
    result = np.sum(hypotheses_minus_price_vector**2) / (2*m)
    return result




C, A, B= 0,0,0
a=0.00001
plt.ion()

xx, yy = np.meshgrid(np.linspace(-5, 5, num=10), np.linspace(-5, 5, num=10))
z = C+A*xx+B*xx
plane = plot.ax.plot_surface(xx, yy, z, color=(0.3,0.7,1,0.5),shade=False) 

# NORMAL EQUATION CALCULATIONS
# plotting normal equation plane
def plot_plane(C,A,B):
    xx, yy = np.meshgrid(np.linspace(-5, 5, num=10), np.linspace(-5, 5, num=10))
    z = C+A*xx+B*yy
    plot.ax.plot_surface(xx, yy, z, color=(1,0.6,0.6,0.5),shade=False) 

# finding global minima
def find_min(X,y):
     X_trasnposed = np.transpose(X)
     theta_vector = np.dot(inv(np.dot(X_trasnposed,X)),np.dot(X_trasnposed,y))
     return theta_vector

# calculating glob min
CC,AA,BB = find_min(X,y) 
print(f'Norm Equations method: '
       f'C {CC*std_price+mean_price:.6f} '
       f'A {AA*std_sqft_living+mean_sqft_living:.6f} '
       f'B {BB*std_age+mean_age:.6f} , SSE {SSE(scaled_np_dps, CC,AA,BB):.6f}\n')

plot_plane(CC,AA,BB)
# NORMAL EQUATION CALCULATIONS END


def redraw_plane(plane, C, A, B):
    plane.remove()
    xx, yy = np.meshgrid(np.linspace(-5, 5, num=10), np.linspace(-5, 5, num=10))
    z = C+A*xx+B*yy
    plane = plot.ax.plot_surface(xx, yy, z, color=(0.3,0.7,1,0.5),shade=False) 
    return plane
    

# GRAD DESCENT CALCULATIONS
for i in range(500000):
    new_grad = grad(scaled_np_dps, C, A, B)
    oldC, oldA, oldB = C, A, B
    C, A, B = np.array([C,A,B]) - a*new_grad  
    if not i % 10000:
        plane = redraw_plane(plane, C,A,B)
        plt.pause(0.001)
        print(f'{f"[{i}]":<6} '
                f'C {C*std_price+mean_price:.6f} '
                f'A {A*std_sqft_living+mean_sqft_living:.6f} '
                f'B {B*std_age+mean_age:.6f} , SSE {SSE(scaled_np_dps, C,A,B):.6f}')

plt.pause(30)
