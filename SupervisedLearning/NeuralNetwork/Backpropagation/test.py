'''
Created on Sep 19, 2016

@author: zaiyong
'''

from numpy import array
from Backpropagation import Backpropagation

if __name__=="__main__":
    dataset=[]
    with open("wine_binary.txt") as lines:
        for line in lines:
            tags=line.split(",")
            dataset.append((array([float(i) for i in tags[1:]]),int(tags[0])))
    
    backpropagation=Backpropagation(dataset,2000)
    backpropagation.trainClassifier()