import numpy as np
import matplotlib.pyplot as plt
from plot2d import Plot2D
from dps_to_nparray import row_datapoints
from math import sqrt, floor

f1 = row_datapoints[:,0] / 250
yy = row_datapoints[:,1] / 250

f0 = np.array([1]*len(f1))
f2 = f1**2 
f3 = f1**3 
f4 = f1**4 




plot = Plot2D((1,1))
plot.scatter0(np.column_stack((f1,yy)))

prepared_data = np.column_stack((f0,f1,f2,f3,f4,yy))

# EFC - Error Function Coefficients
# efc_vector in this case np.array([E,A,B,C,D,-1]) or [ bias, features..., -1 to multiply with y`s ]

def grad(datapoints, efc_vector):
    m = datapoints.shape[0] # amount of rows in matrix    
    errors_vector = datapoints.dot(efc_vector) # hypothesis - real y`s from data
    # ↓ Clever way to multiply errors_vector by each feature in training data 
    # ↓ for each partial derivative and then sum this all up, 
    # ↓ and return each partial derivative in gradient vector 
    # ↓ and then elementwise divide gradient vector by the number of training examples, 
    # ↓ to get real gradient for SSE function:
    grad = np.transpose(datapoints[:,0:5]).dot(errors_vector) / m
    return grad

def SSE(datapoints, efc_vector):
    m = datapoints.shape[0] # amount of rows in matrix
    errors_vector = datapoints.dot(efc_vector)
    result = np.sum(errors_vector**2) / (2*m)
    return result

efc_vector = np.array([0,0,0,0,0,-1])
# param_vector is efc_vector[:,0:-1] or [E,A,B,C,D]
param_vector = efc_vector[0:-1]

a = 1 # fails to converge at f.e. 1.667079

y_neg_component = np.array([-1])

# func does this func(x) = E*1 + A*x + B*x**2 + C*x**3 + D*x**4  or func(x) = [E,A,B,C,D] * [1,x,x**2,x**3,x**4]
func = lambda x: param_vector.dot(np.array([1,x,x**2,x**3,x**4]))

iterations = 10000
for i in range(iterations):
    new_grad = grad(prepared_data, efc_vector)

    old_efc_vector = efc_vector
    old_param_vector = param_vector

    param_vector = old_param_vector - a*new_grad
    efc_vector = np.concatenate((param_vector, y_neg_component))
    sqfnc = lambda x: (x*2)**2

    rendering_iterations = floor(sqrt(iterations/2)) # making rendering steps with quadratic function, to show more useful information
    if i in map(sqfnc, range(rendering_iterations)):
        plot.redraw_func0(func)
        if plot.last_point_func1:
            last_point_x_sse = i - plot.last_point_func1[0]
        else:
            last_point_x_sse = 0            
        plot.add_line_func1(SSE(prepared_data, efc_vector), last_point_x_sse)        
        plt.pause(0.001)
        print(i, param_vector)
plt.ioff()
plt.show()
input()