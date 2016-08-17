'''
Created on Aug 17, 2016

@author: zaiyong
'''
from NaiveBayesClassifier import NaiveBayesClassifier
import csv
from scipy.constants.codata import precision

if __name__=="__main__":
    lines = csv.reader(open("test.csv", "rb"))
    dataset = []
    for line in lines:
        dataset.append([float(x) for x in line])
    clssifier=NaiveBayesClassifier(dataset)
    trainset,testset=clssifier.spliteTrainTest()
    clssifier.createClassifier(trainset)
    tp=0
    fp=0
    tn=0
    fn=0
    for case in testset:
        predicted_class=clssifier.predict(case[:-1])
        if predicted_class==1:
            if case[-1]==1: tp+=1
            else: fp+=1
        else:
            if case[-1]==1: fn+=1
            else:tn+=1
    accuracy=float(tp+tn)/len(testset)
    F1=2*tp/float(2*tp+fp+fn)
    recall=tp/float(tp+fn)
    precision=tp/float(tp+fp)
    print("accuray: %f\nF1: %f\nrecall: %f\nprecision: %f"%(accuracy,F1,recall,precision))