import numpy as np
from datapoints import set_seed, labeled_Points
from plot2d import Plot2D
from polyclf import PolyClassifier

#HYPER PARAMETERS
#lambda
l = 0.3
#λ = -0.1
# learning rate
a = 0.3
#polynomial degree
degree = 40
# step before redrawing
step = 100
# computing iterations
iterations = 6000
#random seed
set_seed(29) 
#Video TITLE
title = f'Polynomial Logistic Regression with regularization, λ={l}'
#Video format
video_fmt = 'gif' #mp4
#Video filename
video_filename = f'polyclf_regular_lambda_eq_{l}'

datapoints = labeled_Points(Aweight=4, Bweight=2, 
                            Aamount=25, Bamount=12, 
                            scale=10)

plot = Plot2D((10,10), color='white', suptitle=title, 
              video=True, vide_fmt='gif', 
              video_filename=video_filename)

plot.plot_labeled_data(datapoints)
clf = PolyClassifier(datapoints, a=a, l=l, degree=degree)


try:
    for i in range(1,iterations+1):
        clf.gradient_step()
        if not i % step:            
            plot.plot_decision_surface(clf.predictor_func)
            cost = clf.CostFunction()
            print(f'Theta is: {clf.theta}')
            print(f'[{i:08}] Cost is: {cost}')
            
            #print(f'  Theta: {clf.theta}')
            text = ( f'λ = {l}\n'
                     f'α = {a}\n'
                     f'Degree = {degree}\n'
                     f'Cost = {cost:.8f}\n'
                     f'Itter = {i:08}'      )
            plot.add_line_cost_func(cost,1, text=text)
            plot.pause(0.01)
            plot.add_frame_to_video() 
finally:
    plot.write_video_to_storage()

# with open('theta.py', 'w+') as f:
#     f.write('import numpy as np\n')
#     f.write(f'theta = np.array({list(clf.theta)})')


            



    