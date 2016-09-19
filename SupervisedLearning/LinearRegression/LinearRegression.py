'''
Created on Aug 18, 2016

@author: zaiyong
'''
import numpy as np

class LinearRegression:
    def __init__(self,dataset):
        self.dataset=np.mat(dataset)
        self.X=self.dataset[:,0:-1]
        self.y=self.dataset[:,-1]

    def simpleLeastSquares(self):
        X=np.insert(self.X,0,1,axis=1)
        X1=np.matrix.transpose(X)*X
        return np.linalg.inv(X1)*np.matrix.transpose(X)*self.y
