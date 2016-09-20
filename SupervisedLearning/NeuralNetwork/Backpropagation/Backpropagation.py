'''
Created on Sep 20, 2016

@author: zaiyong
'''
import numpy,random

class Backpropagation:
    def __init__(self,dataset,iterate_number,eta=0.1):
        self.dataset=dataset
        self.iterate_number=iterate_number
        self.eta=eta
        numpy.random.shuffle(self.dataset)
        self.data_size=len(self.dataset)
        self.input_node_num=len(self.dataset[0][0])
        self.hidden_node_num=len(self.dataset[0][0])
        self.output_node_num=1

    def spliteTrainTest(self,split_ratio=0.67):
        train_size=int(self.data_size*split_ratio)
        self.train_input=[]
        self.train_expect=[]
        self.test_input=[]
        self.test_expect=[]
        for i,e in self.dataset[0:train_size]:
            self.train_input.append(i)
            self.train_expect.append(e)

        for i,e in self.dataset[train_size:]:
            self.test_input.append(i)
            self.test_expect.append(e)

        self.train_input=numpy.matrix(self.train_input)
        self.train_expect=numpy.matrix(self.train_expect)
        self.test_input=numpy.matrix(self.test_input)
        self.test_expect=numpy.matrix(self.test_expect)
        

        
    def initWeights(self):
        self.weights_input_hidden=numpy.random.rand(self.input_node_num,self.hidden_node_num)
        self.weights_hidden_output=numpy.random.rand(self.hidden_node_num,self.output_node_num)
    
    
    def initBias(self):
        self.bias_hidden=numpy.zeros(self.hidden_node_num)
        self.bias_output=numpy.zeros(self.output_node_num)
    
    def hiddenValues(self,input):
        return numpy.tanh(input*self.weights_input_hidden+self.bias_hidden)
    
    def outputValues(self,hidden):
        return numpy.transpose(hidden*self.weights_hidden_output+self.bias_output)
    
    def error(self,predict,expect):
        print (expect-predict)
        return numpy.multiply((expect-predict),(expect-predict))/2.0
    
    def updateWeights(self,input,hidden,predict,expect):
        error=expect-predict
        print error.shape
        for i in range(self.input_node_num):
            for j in range(self.hidden_node_num):
                self.weights_input_hidden[i,j]=self.weights_input_hidden[i,j]+self.eta*hidden[i,j]*(1-hidden[i,j])*input[i,j]*sum([error[0][i]*w for w in self.weights_hidden_output[j]])
        
        for j in range(self.hidden_node_num):
            for k in range(self.output_node_num):
                self.weights_hidden_output[j,k]=self.weights_hidden_output[j,k]+self.eta*hidden[j]*error[0][j]
    
    def updateBias(self,hidden,predict,expect):
        error=expect-predict
        for j in range(self.hidden_node_num):
            self.bias_hidden[j]=self.bias_hidden[j]+self.eta*hidden[j]*(1-hidden[j])*numpy.dot(self.weights_hidden_output[j],error)
        for k in range(self.output_node_num):
            self.bias_output[k]=self.bias_output[k]+self.eta*error[k]
            
    def trainClassifier(self):
        self.spliteTrainTest()
        self.initWeights()
        self.initBias()
        for i in range(self.iterate_number):
            print i
            hidden_values=self.hiddenValues(self.train_input)
            print hidden_values.shape
            output_values=self.outputValues(hidden_values)
            print "abc",output_values
            print "bcd",self.train_expect
            self.updateWeights(self.train_input, hidden_values, output_values, self.train_expect)
            self.updateBias(hidden_values, output_values, self.train_expect)

            
        
            
            
            
        