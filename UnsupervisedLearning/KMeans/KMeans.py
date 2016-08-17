'''
Created on Aug 16, 2016

@author: zaiyong
'''
import numpy as np

class kMeans:
    def __init__(self,dataset,k):
        self.dataset=dataset
        self.k=k
        self.x_dim,self.y_dim=self.dataset.shape
    def calculateEuclideanDistance(self,point1,point2):
        return np.linalg.norm(point1-point2)
    
    def initCluster(self):  
        centroids =[]
        choices=np.random.choice(range(0,self.x_dim),self.k,False)
        for i in choices:    
            centroids.append( self.dataset[i])   
        cluster=[]
        for point in self.dataset:
            cluster.append([point,self.findClosestCentroid(point, centroids)])
        return centroids,cluster
    
    def findClosestCentroid(self,point,centroids):
        minDistance=1000000.00
        closest_centroid=centroids[0]
        for i in range(0,self.k):
            distance=self.calculateEuclideanDistance(point, centroids[i])
            if distance<minDistance: 
                minDistance=distance
                closest_centroid=centroids[i]
        return closest_centroid
    
    def updateCluster(self,centroids,cluster):
        new_centroid=[]
        for i in centroids:
            new_centroid.append(np.mean([ a[0] for a in cluster if (a[1]==i).all()],0))
        return new_centroid
    
    def kMeans(self):
        cluster_changed=True
        centroids,cluster=self.initCluster()
        centroids=self.updateCluster(centroids, cluster)
        N=0
        while cluster_changed:
            N+=1
            cluster_changed=False
            for i in range(0,self.x_dim):
                resign_centroid=self.findClosestCentroid(cluster[i][0], centroids)
                if  not (resign_centroid==cluster[i][1]).all():
                    cluster_changed=True
                    cluster[i][1]=resign_centroid
            centroids=self.updateCluster(centroids, cluster)
        
        return centroids,cluster
