#y y2
#z z2

from numpy import maximum, minimum, abs, exp, linspace, log
import numpy as np
import matplotlib.pyplot as plt



z = linspace(-5, 5, 11)

sigma = lambda x: 1 /(1 + exp(-x))
f = lambda z: -log(sigma(z))
f2 = lambda z: -log(1-sigma(z))
f11 = lambda z: abs(minimum(z-1,0))
f22 = lambda z: maximum(z+1,0)

y = f(z)
y2 = f2(z)
y11 = f11(z)
y22  = f22(z)


fig = plt.figure(figsize=(16,8), dpi=100)
fig.suptitle('Loss function J(θ)\nlogistic regression vs svm')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.spines['bottom'].set_position('center')
ax1.spines['left'].set_position('center')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.spines['bottom'].set_position('center')
ax2.spines['left'].set_position('center')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax1.set_title('y=1')
ax2.set_title('y=0')

ax1.axis([-3,3,-5,5])
ax2.axis([-3,3,-5,5])

ax1.plot(z,y, 'g-' , label='J Lin. Reg.')
ax1.plot(z,y11, 'r-', label='J SVM')
ax2.plot(z,y2, 'g-' , label='J Lin. Reg.')
ax2.plot(z,y22, 'r-', label='J SVM')

ax1.legend()
ax2.legend()

plt.show()
