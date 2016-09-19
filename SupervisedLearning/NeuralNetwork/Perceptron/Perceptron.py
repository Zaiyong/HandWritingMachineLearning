'''
Created on Sep 19, 2016

@author: zaiyong
'''

from random import choice
from numpy import dot, random

class Perceptron:
    def __init__(self,dataset,iterate_number):
        self.dataset=dataset
        self.iterate_number=iterate_number
        random.shuffle(self.dataset)
        self.data_size=len(self.dataset)

    def spliteTrainTest(self,split_ratio=0.67):
        train_size=int(self.data_size*split_ratio)
        self.train_set=self.dataset[0:train_size]
        self.test_set=self.dataset[train_size:]
    
    def initWeights(self):
        self.weights=random.rand(len(self.dataset[0][0]))
    
    def decision(self,value):
        return 1 if value >0 else 0
        
    def trainClassifier(self):
        self.initWeights()
        self.spliteTrainTest()
        errors=[]
        eta=0.1
        for i in range(self.iterate_number):
            input,target=choice(self.train_set)
            result = dot(self.weights, input)
            error = target - self.decision(result)
            self.weights += eta * error * input

    def test(self):
        tp=0
        fp=0
        tn=0
        fn=0
        for input,target in self.test_set:
            predict=self.decision(dot(input,self.weights))
            if predict==1:
                if target==1: tp+=1
                else: fp+=1
            else:
                if target==1: fn+=1
                else:tn+=1
        accuracy=float(tp+tn)/len(self.test_set)
        F1=2*tp/float(2*tp+fp+fn)
        recall=tp/float(tp+fn)
        precision=tp/float(tp+fp)
        print("accuray: %f\nF1: %f\nrecall: %f\nprecision: %f"%(accuracy,F1,recall,precision))