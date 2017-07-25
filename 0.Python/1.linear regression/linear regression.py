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
fig.suptitle('SquareFootage and Price', fontsize=14, fontweight='bold')
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_title('SquareFootage and Price')
ax.title.set_color((0.7,0,0))

ax.spines['bottom'].set_color((1,1,1))
# ax.spines['top'].set_color((1,1,1))
# ax.spines['right'].set_color((1,1,1))
ax.spines['left'].set_color((1,1,1))
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

ax.set_facecolor('black')
ax.set_xlabel('SquareFootage')
ax.set_ylabel('Price')
ax.yaxis.label.set_color((1,1,1))
ax.xaxis.label.set_color((1,1,1))


ax.axis([0,1,0,1])
plt.ion()
price_arrayy = []
sqft_arrayy = []

with open('houseprices.csv', 'r') as datafile:
    csv_dict = csv.DictReader(datafile)
    for row in csv_dict:
        datapoint = filter_float_tupples(row['price'], row['sqft_living'])        
        if datapoint:
            price_arrayy.append(datapoint[0] / 10000000)
            sqft_arrayy.append(datapoint[1] / 15000)

print(f'max price is {max(price_arrayy)}')
print(f'min price is {min(price_arrayy)}')

print(f'max sqft is {max(sqft_arrayy)}')
print(f'min sqft is {min(sqft_arrayy)}')

price_array = price_arrayy
sqft_array = sqft_arrayy

price_array = [ price + 0.3 for price in price_array]
m = 0
b = 0
x = np.linspace(0, 15000, num=2)
func = lambda x: x*m+b
y = np.fromiter(map(func,x), dtype = np.float)

lines = plt.plot(x,y, color='red')

plt.scatter(sqft_array, price_array, s=0.3)
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

bias_array = [ 1 ] * len(sqft_array)
datapoints = np.array(list(zip(bias_array, sqft_array, price_array)))
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
am=1
ab=0.1


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
    if b-old_b > 0.0001 or m-old_m > 0.0001 or not i % 5000:
        lines = redraw_line(lines, b, m)
        plt.pause(0.001)
        print(f'{f"[{i}]":<6} B {b:.6f} M {m:.6f}, SSE {SSE(datapoints, b,m):.6f}')
    




    


