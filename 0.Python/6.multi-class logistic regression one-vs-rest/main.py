import numpy as np
from datapoints import set_seed, labeled_three_classPoints
from plot2d import Plot2D
from mcclf import OnevsRestClf

set_seed(143)
datapoints = labeled_three_classPoints(Aweight=5, Bweight=3, Cweight=2,
                                     Aamount=500, Bamount=500, Camount=500,
                                     scale=10)
plot = Plot2D((10,10))


plot.plot_labeled_data(datapoints)


clf = OnevsRestClf(datapoints, a=0.03)

for i in range(1000000):
    clf.gradient_step()
    if not i % 100:
        plot.plot_decision_surface(clf.predictor_func)
        cost = clf.CostFunction()
        print(f'[{i:07}] Cost is: {cost}')
        plot.add_line_cost_func(cost,1)
        plot.pause(0.01)


    