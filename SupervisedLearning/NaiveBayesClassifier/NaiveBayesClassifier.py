'''
Created on Aug 17, 2016

@author: zaiyong
'''
import numpy as np
import scipy.stats

class NaiveBayesClassifier:
    def __init__(self,dataset):
        self.dataset=dataset
        np.random.shuffle(self.dataset)
        self.data_size=len(self.dataset)
        self.classifier = {}
        self.class_probability={}
        self.classes=[]

    def spliteTrainTest(self,split_ratio=0.67):
        train_size=int(self.data_size*split_ratio)
        return self.dataset[0:train_size],self.dataset[train_size:]
    
    def separateByClass(self,trainset):
        separated = {}
        for i in range(len(trainset)):
            case= trainset[i]
            if (case[-1] not in self.classes):
                separated[case[-1]] = []
                self.classes.append(case[-1])
            separated[case[-1]].append(case[:-1])
        return separated
        
    def createClassifier(self,trainset):
        separated = self.separateByClass(trainset)
        for _class, cases in separated.iteritems():
            self.classifier[_class]=[]
            self.class_probability[_class]=len(cases)/float(len(trainset))
            matrix=np.mat(cases)
            for i in range(matrix.shape[1]):
                self.classifier[_class].append([np.mean(matrix[:,i]),np.std(matrix[:,i])])
    
    def calculateProbabilities(self,case):
        probabilities={}
        for _class in self.classes:
            probabilities[_class]=self.class_probability[_class]
            for i in range(len(case)):
                m=self.classifier[_class][i][0]
                std=self.classifier[_class][i][1]
                probabilities[_class]*=scipy.stats.norm(m,std).pdf(case[i])
        return probabilities
        