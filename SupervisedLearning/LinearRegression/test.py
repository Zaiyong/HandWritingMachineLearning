'''
Created on Aug 18, 2016

@author: zaiyong
'''
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from LinearRegression import LinearRegression
import numpy as np

if __name__=="__main__":
    dataset=[]
    for line in open("test.txt").readlines():
        tags=line.strip().split(",")
        dataset.append([float(x) for x in tags])
        
    regression=LinearRegression(dataset)
    B=regression.simpleLeastSquares()
    fig=pyplot.figure()
    ax = fig.gca( projection='3d')
    ax.plot([0,max(np.mat(dataset)[:,0])[0,0]],
            [0,max(np.mat(dataset)[:,1])[0,0]],
            [([1,0,0]*B)[0,0],([1,max(np.mat(dataset)[:,0]),max(np.mat(dataset)[:,1])]*B)[0,0]],label="test")
    ax.scatter([point[0] for point in dataset],[point[1] for point in dataset],[point[2] for point in dataset])
    #pyplot.plot([0,800],[b[0,0],b[0,0]+800*b[1,0]])
    #pyplot.plot(point[0],point[1],"or")
    pyplot.show()