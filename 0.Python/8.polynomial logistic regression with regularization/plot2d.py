import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
from matplotlib.ticker import MaxNLocator
from datapoints import labeled_Points, set_seed



class Plot2D:
    def __init__(self, XY, color='black', suptitle='', video=False, vide_fmt='gif', video_filename='noname_animation'):
        assert color == 'black' or color == 'white', 'Only black and white supported'
        if color =='black':
            self.colors = [ (0.81,  0.28 ,  0.29),
                            (0.53,  0.90 ,  0.99),
                            (0,     0.86 ,  0.42),
                            (0.91,  0.55 ,  0.97),
                            (1,     0.75 ,  0   ) ]
        else:
            self.colors = [ (0.51,  0.18 ,  0.19),
                            (0.13,  0.50 ,  0.89),
                            (0,     0.86 ,  0.42),
                            (0.91,  0.55 ,  0.97),
                            (1,     0.75 ,  0   ) ]


        self.color = color
        self.xy = XY
        self.last_point_func1 = None
        self.lines_func0 = None
        self.lines_func1 = None
        self.linspace0 = np.linspace(-1*self.xy[0], self.xy[0], num=100)
        mpl.rcParams['toolbar'] = 'None'
        self.fig = plt.figure(figsize=(12, 6), dpi=100)
        if self.color == 'black':
            self.fig.set_facecolor((0,0,0))
            self.fig.suptitle(suptitle, fontsize=14, color=(0,0.6,0))
        else:
            self.fig.set_facecolor((1,1,1))
            self.fig.suptitle(suptitle, fontsize=14, color=(0,0,0))
        self.ax0 = self.fig.add_subplot(121)
        self.ax1 = self.fig.add_subplot(122)
        self.fig.subplots_adjust(top=0.85)

        if self.color == 'black':
            self.ax0.set_facecolor('black')
            self.ax0.set_title('Decision surface', color='white')
            self.ax0.spines['bottom'].set_color((1,1,1))
            self.ax0.spines['left'].set_color((1,1,1))
            self.ax0.tick_params(axis='x', colors='white')
            self.ax0.tick_params(axis='y', colors='white')
        else:
            self.ax0.set_facecolor('white')  
            self.ax0.set_title('Decision surface', color='black')
            self.ax0.spines['bottom'].set_color((0,0,0))
            self.ax0.spines['left'].set_color((0,0,0))          
            self.ax0.tick_params(axis='x', colors='black')
            self.ax0.tick_params(axis='y', colors='black')

        self.ax0.spines['bottom'].set_position('center')
        self.ax0.spines['left'].set_position('center')
        self.ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

        self.ax0.axis([-1*self.xy[0],
                        self.xy[0],
                       -1*self.xy[1],
                        self.xy[0]])
        
        #hide 0 ticks
        xticks = self.ax0.xaxis.get_major_ticks()
        xticks[3].label1.set_visible(False)
        
        yticks = self.ax0.yaxis.get_major_ticks()
        yticks[3].label1.set_visible(False)
        
        if self.color == 'black':            
            self.ax1.set_facecolor('black')
            self.ax1.set_title('J(θ)', color='white')
            self.ax1.spines['bottom'].set_color((1,1,1))
            self.ax1.spines['left'].set_color((1,1,1))
            
            self.ax1.tick_params(axis='x', colors='white')
            self.ax1.tick_params(axis='y', colors='white')
        else:
            self.ax1.set_facecolor('white')
            self.ax1.set_title('J(θ)', color='black')
            self.ax1.spines['bottom'].set_color((0,0,0))
            self.ax1.spines['left'].set_color((0,0,0))
            
            self.ax1.tick_params(axis='x', colors='black')
            self.ax1.tick_params(axis='y', colors='black')



        # plt.legend(loc='upper left', scatterpoints=1, numpoints=1)

        self.contours  = None
        self.contourln = None
        self.labels    = None
        self.cost_text = None

        self.video = video
        self.video_fmt = vide_fmt
        self.video_filename = video_filename
        self.metadata = dict(title=suptitle, artist='Artemenko EV')
        self.extension = 'gif' if self.video_fmt == 'gif' else 'mp4'
        if self.video_fmt == 'gif':
            self.writer = manimation.ImageMagickWriter(fps=25, metadata=self.metadata)
        elif self.video_fmt == 'mp4':
            self.writer = manimation.FFMpegWriter(fps=25, metadata=metadata)
        self.writer.setup(self.fig, f'{self.video_filename}.{self.video_fmt}',  dpi=120)

        self.show()


    def add_frame_to_video(self):
        if self.video:
            self.writer.grab_frame(facecolor=self.color)

    def write_video_to_storage(self):
        if self.video:
            self.writer.finish()

    def show(self):
        plt.ion()


    def plot_decision_surface(self, predictor_func):
        self.clear(contours=True)

        x1_min = -self.xy[0]
        x1_max =  self.xy[0]
        x2_min = -self.xy[1]
        x2_max =  self.xy[1]
        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                             np.arange(x2_min, x2_max, 0.1))
        Y = predictor_func(np.c_[x1.ravel(),x2.ravel()])
        Y = Y.reshape(x1.shape)
        cmap = 'Set1' if self.color == 'black' else 'RdBu'
        self.contours = self.ax0.contourf(x1, x2, Y, cmap=cmap, alpha=0.30)
        levels = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
        color = 'white' if self.color == 'black' else 'black'
        self.contourln =  self.ax0.contour(self.contours, levels=levels,
                                           colors=color, linewidths=(1,), origin='lower')        
        self.labels = self.ax0.clabel(self.contourln, fmt='%.2f', colors=color, fontsize=10)
        print(type(self.labels))


    def clear(self, contours=False, cost_text=False):
        if contours and self.contours:
            for c in self.contours.collections:
                c.remove()
        if contours and self.contourln:
            for cln in self.contourln.collections:  
                cln.remove()
            for l in self.labels:
                l.remove()
        if cost_text and self.cost_text:
           self.cost_text.remove()



    def plot_labeled_data(self, datapoints):
        # where datapoints is a list of numbers (x1, x2,  y) 
        # y is label (1 or 0)
        # x1, x2 is features
        assert type(datapoints) is np.ndarray, 'datapoint should be numpy.array'
        assert datapoints.shape[1] == 3, 'datapoint format should be (x1, x2 ,y)'
        
        y = np.rint(datapoints[:,2]).astype(int)
        assert max(y) <= 4 or min(y) >= 0, 'Plotter can accept only up to 5 classes, labled from 0 to 4'

        classes = list(set(y))
        for c in classes:
            class_points = np.array([ row[0:2] for row in datapoints 
                                                   if np.round(row[2]) == c])
            self.ax0.scatter(class_points[:,0],class_points[:,1], s=50, c=self.colors[c], 
                             marker='o', label=f'Class{c}')


    def pause(self, sec):
        plt.pause(sec)


    def add_line_cost_func(self, val, step, text=''):
        self.clear(cost_text=True)
        self.cost_text = self.ax1.text(0.5, 0.5, text,
                                       verticalalignment='center',
                                       horizontalalignment='center',
                                       transform=self.ax1.transAxes,
                                       color='black', fontsize=12)
        # from self.last_point_func1 = 0
        if not self.last_point_func1:
            self.last_point_func1 = (0,val)
        p1 = self.last_point_func1
        p2 = (self.last_point_func1[0]+step, val)
        color = 'white' if self.color == 'black' else 'black'
        self.ax1.plot( (p1[0],p2[0]), (p1[1],p2[1]), color=color)

        # moving x by step and y making = val
        self.last_point_func1 = p2
        


if __name__ == '__main__':
    plot = Plot2D((10,10))
    set_seed(81) #137, 131, 124, 123, 119, 106, 85, 82, 80, 67, 56, 53
    datapoints = labeled_Points(Aweight=3, Bweight=2, Aamount=700, Bamount=700, scale=10)
    print(datapoints.shape)
    plot.plot_labeled_data(datapoints)
    plot.pause(1)
    from sklearn.svm import SVC


    clf =  SVC(kernel='rbf', probability=True)

    X = datapoints[:,:-1]
    y = datapoints[:,-1]
    
    clf.fit(X,y)

    
    plot.plot_decision_surface(clf.predict)
    plot.pause(1)


    #plot.clear(contours=True)



    from math import log
    for i in range(1,10000):    
        print(i)
        f = lambda x: 300/x**(1/2)
        plot.add_line_cost_func(f(i),0.1)
        plot.pause(0.0001)

    plot.pause(100)









