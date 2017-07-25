import warnings
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv



class Plot2D:
    def __init__(self, XY):
        self.xy = XY
        self.last_point_func1 = None
        self.lines_func0 = None
        self.lines_func1 = None
        self.linspace0 = np.linspace(-1*self.xy[0], self.xy[0], num=100)
        mpl.rcParams['toolbar'] = 'None'
        self.fig = plt.figure(figsize=(20, 10), dpi=80)
        self.fig.set_facecolor((0,0,0))
        self.fig.suptitle('Polynomial regression with toy data', fontsize=14, color=(0,0.6,0))
        self.ax0 = self.fig.add_subplot(121)
        self.ax1 = self.fig.add_subplot(122)
        self.fig.subplots_adjust(top=0.85)
        
        
        self.ax0.set_facecolor('black')
        
        self.ax0.spines['bottom'].set_color((1,1,1))
        self.ax0.spines['left'].set_color((1,1,1))
        self.ax0.spines['bottom'].set_position('center')
        self.ax0.spines['left'].set_position('center')
        self.ax0.tick_params(axis='x', colors='white')
        self.ax0.tick_params(axis='y', colors='white')
        self.ax0.axis([-1*self.xy[0],
                        self.xy[0],
                       -1*self.xy[1],
                        self.xy[0]])
        
        #hide 0 ticks
        xticks = self.ax0.xaxis.get_major_ticks()
        xticks[3].label1.set_visible(False)
        
        yticks = self.ax0.yaxis.get_major_ticks()
        yticks[3].label1.set_visible(False)
        
        
        self.ax1.set_facecolor('black')
        self.ax1.spines['bottom'].set_color((1,1,1))
        self.ax1.spines['left'].set_color((1,1,1))
        
        self.ax1.tick_params(axis='x', colors='white')
        self.ax1.tick_params(axis='y', colors='white')

        self.show()


    def show(self):
        plt.ion()



    def redraw_func0(self, func):
        # removing old lines
        if self.lines_func0:
            for i in range(len(self.lines_func0)):
                self.lines_func0.pop(i).remove()
        # as x_array in __init__ defined self.linspace0
        y_array = np.fromiter(map(func,self.linspace0), dtype = np.float)
        self.lines_func0 = self.ax0.plot(self.linspace0, y_array, color='red')


    def scatter0(self, datapoints):
        assert type(datapoints) is np.ndarray, 'datapoint should be numpy.array'
        self.ax0.scatter(datapoints[:,0],datapoints[:,1], s=50, c='g', marker='x')


    def add_line_func1(self, val, step):
        # from self.last_point_func1 = 0
        if not self.last_point_func1:
            self.last_point_func1 = (0,val)
        p1 = self.last_point_func1
        p2 = (self.last_point_func1[0]+step, val)

        self.ax1.plot( (p1[0],p2[0]), (p1[1],p2[1]), color='white')

        # moving x by step and y making = val
        self.last_point_func1 = p2
        


if __name__ == '__main__':
    plot = Plot2D((5,5))

    # test scatter
    datapoints = np.array([[1,1],
                           [2,2],
                           [3,3]])
    plot.scatter0(datapoints)
     
    #  Test first and second func
    from math import sin, cos
    for i in range(10000):
        f = lambda x: sin(x+i*0.05)
        plot.redraw_func0(f)
        
        print(i)
        f2 = lambda x: sin(i)
        plot.add_line_func1(cos(i*0.1),0.1)
        if  not i % 100:
            plt.pause(0.0001)





