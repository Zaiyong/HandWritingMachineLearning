'''
Created on Aug 16, 2016

@author: zaiyong
'''
import numpy as np
class kMeans:
    def __init__(self,dataset,k):
        self.dataset=dataset
        self.k=k
        
    def calculateEuclideanDistance(self,point1,point2):
        return np.sqrt(sum(np.power(point1-point2, 2)))
    
    def initCentroids(self):  
        x_dim, y_dim = self.dataset.shape  
        centroids = np.zeros((self.k, y_dim))  
        for i in range(self.k):  
            index = int(self.random.uniform(0, x_dim))  
            centroids[i, :] = self.dataset[index, :]  
        return centroids