import numpy as np
from datapoints import set_seed, labeled_Points
from plot2d import Plot2D
from polyclf import PolyClassifier

set_seed(106) #137, 131, 124, 123, 119, 106, 85, 82, 80, 67, 56, 53
datapoints = labeled_Points(Aweight=11, Bweight=2, 
                            Aamount=500, Bamount=500, 
                            scale=10)
plot = Plot2D((10,10))


plot.plot_labeled_data(datapoints)


clf = PolyClassifier(datapoints, a=0.01)

for i in range(1000000):
    clf.gradient_step()
    if not i % 1000:
        plot.plot_decision_surface(clf.predictor_func)
        cost = clf.CostFunction()
        print(f'[{i:07}] Cost is: {cost}')
        print(f'  Theta: {clf.theta}')
        plot.add_line_cost_func(cost,1)
        plot.pause(0.01)


    