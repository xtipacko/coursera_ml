import numpy as np
from datapoints import set_seed, labeled_Points
from plot2d import Plot2D
from polyclf import PolyClassifier
from theta import theta
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Polynomial Regression', artist='Matplotlib',
                comment='with initial normalization!')
writer = FFMpegWriter(fps=24, metadata=metadata)

set_seed(24) #137, 131, 124, 123, 119, 106, 85, 82, 80, 67, 56, 53
datapoints = labeled_Points(Aweight=15, Bweight=2, 
                            Aamount=8000, Bamount=2000, 
                            scale=10)
plot = Plot2D((10,10))


plot.plot_labeled_data(datapoints)


clf = PolyClassifier(datapoints, a=45, degree=18)
clf.theta = theta
step = 10000
with writer.saving(plot.fig, "polynomial_regression__in.mp4", 100):
    for i in range(10000000):
        clf.gradient_step()
        if not i % step:            
            plot.plot_decision_surface(clf.predictor_func)
            cost = clf.CostFunction()
            print(f'Theta is: {clf.theta}')
            print(f'[{i:08}] Cost is: {cost}')
            
            #print(f'  Theta: {clf.theta}')
            plot.add_line_cost_func(cost,1)
            plot.pause(0.01)
            writer.grab_frame()
with open('theta.py', 'w+') as f:
    f.write('import numpy as np\n')
    f.write(f'theta = np.array({list(clf.theta)})')


            



    