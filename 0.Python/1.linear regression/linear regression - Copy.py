from time import sleep
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
def filter_float_tupples(str_price, str_sqft):
    try:
        price = float(str_price)
        sqft = float(str_sqft)
        return (price, sqft)
    except:
        return None




matplotlib.rcParams['toolbar'] = 'None'
fig = plt.figure()
fig.set_facecolor((0,0,0))
fig.suptitle('Heat and number of atoms', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_title('Heat and number of atoms')
ax.title.set_color((0.7,0,0))

ax.spines['bottom'].set_color((1,1,1))
# ax.spines['top'].set_color((1,1,1))
# ax.spines['right'].set_color((1,1,1))
ax.spines['left'].set_color((1,1,1))
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

ax.set_facecolor('black')
ax.set_xlabel('Heat')
ax.set_ylabel('number of atoms')
ax.yaxis.label.set_color((1,1,1))
ax.xaxis.label.set_color((1,1,1))



plt.ion()
heat_array = []
molec_array = []

with open('chem.csv', 'r') as datafile:
    csv_dict = csv.DictReader(datafile)
    for row in csv_dict:
        heat_array.append(float(row['heat']))
        molec_array.append(float(row['molecules']))

print(f'max heat is {max(heat_array)}')
print(f'min heat is {min(heat_array)}')

print(f'max atoms in molec is {max(molec_array)}')
print(f'min atoms in molec is {min(molec_array)}')
ax.axis([min(molec_array)-3,max(molec_array),min(heat_array),max(heat_array)])

m = -400
b = -2000
x = np.linspace(-2, 10, num=2)
func = lambda x: x*m+b
y = np.fromiter(map(func,x), dtype = np.float)

lines = plt.plot(x,y, color='red')

plt.scatter(molec_array, heat_array, s=3)
plt.pause(1)

def redraw_line(lines, b, m):
    for i in range(len(lines)):
        lines.pop(i).remove()
    func = lambda x: x*m+b
    y = np.fromiter(map(func,x), dtype = np.float)
    lines = plt.plot(x,y, color='red')
    return lines

# plt.pause(1)
# lines = redraw_line(lines, 1000, 0)



# b - feta0, m - feta1
# datapoints = list(zip(sqft_array, price_array))

bias_array = [ 1 ] * len(molec_array)
datapoints = np.array(list(zip(bias_array, molec_array, heat_array)))
l = len(datapoints)

def gradient(datapoints, b, m):

    v = np.array([b,m,-1])
    mv_prod = datapoints.dot(v)
    new_b = 1/l*np.sum(mv_prod)
    new_m = 1/l*np.sum(np.multiply(mv_prod, datapoints[:,1])) # sqft_array in column vector = datapoints[:,1]

    # sqft_array in column vector = datapoints[:,1]
    # np.multiply - componentwise multiplication
    # .dot - dot multiplication
    # new_b = 1/l*sum([ b + m*Xi - Yi      for Xi, Yi in datapoints])
    # new_m = 1/l*sum([ (b + m*Xi - Yi)*Xi for Xi, Yi in datapoints])
    return new_b, new_m

#learning_rates
am=0.02
ab=0.2


def SSE(datapoints, b, m):
    v = np.array([b,m,-1])
    mv_prod = datapoints.dot(v)
    result = 1/(2*l)*np.sum(np.multiply(mv_prod, mv_prod))
    return result


for i in range(1000000):
    grad_b, grad_m = gradient(datapoints, b, m)
    # for visualisation
    old_b = b
    old_m = m
    #
    b = b - ab*grad_b
    m = m - am*grad_m
    if b-old_b > 0.01 or m-old_m > 0.01 or not i % 5000:
        lines = redraw_line(lines, b, m)
        plt.pause(0.001)
        print(f'{f"[{i}]":<6} B {b:.6f} M {m:.6f}, SSE {SSE(datapoints, b,m):.6f}')

print(f'{f"[{i}]":<6} B {b:.6f} M {m:.6f}, SSE {SSE(datapoints, b,m):.6f}')
    




    


