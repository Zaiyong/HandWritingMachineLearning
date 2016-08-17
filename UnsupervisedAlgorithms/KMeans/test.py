'''
Created on Aug 17, 2016

@author: zaiyong
'''

from KMeans import kMeans
from matplotlib import pyplot
import numpy as np

if __name__=="__main__":
    dataset=[]
    for line in open("test.dat").readlines():
        tags=line.strip().split()
        dataset.append([float(tags[0]),float(tags[1])])
    data_mat=np.mat(dataset)
    k_means=kMeans(data_mat,4)
    centroids,cluster=k_means.kMeans()
    mark_dataset = ['or', 'ob', 'og', 'ok'] 
    mark_centroid = ['Dr', 'Db', 'Dg', 'Dk']
    i=0
    for centroid in centroids:
        group=[ a[0] for a in cluster if (a[1]==centroid).all()]
        for point in group:
            pyplot.plot(point[0,0],point[0,1],mark_dataset[i],markersize=5)
        pyplot.plot(centroid[0][0],centroid[0][1], mark_centroid[i], markersize = 10)
        i+=1
    pyplot.show()