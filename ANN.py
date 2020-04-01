"""This file holds the Artificial Neural Model"""

import numpy as np

class neural_network:

    def __init__(self, training_dataset, training_targets, lr = 0.2, hidden_size = 10):
        
        self.__training_dataset = training_dataset
        self.__training_targets = training_targets
        self.__lr = lr
        self.__hidden_size = hidden_size        
        
        self.__error = 0.0
        self.__correct_count = 0

        np.random.seed(1)
        
        self.__weights_01 = 0.2 * np.random.random((784, self.__hidden_size)) - 0.1
        self.__weights_12 = 0.2 * np.random.random((self.__hidden_size, len(self.__training_targets[0]))) - 0.1
        
        #Activation Functions
        self.RELU = lambda x : (x >= 0) * x
        self.RELU2DERIV = lambda x : x >= 0

        print '[+] Neural Network Initilized'

    def forward_propagation(self, vector):
        
        layer_1 = vector
        layer_2 = self.RELU(np.dot(layer_1, self.__weights_01))
        layer_3 = np.dot(layer_2, self.__weights_12)
        
        return layer_1, layer_2, layer_3
    
    def backpropagation(self, p):
        
        #Forward Propagate
        layer_1, layer_2, layer_3 = self.forward_propagation(self.__training_dataset[p : p + 1])
        

        #Calculate Error
        
        self.__error += np.sum((self.__training_targets[p : p + 1] - layer_3) ** 2)
        
        
        
        self.__correct_count += int(np.argmax(layer_2)) == np.argmax(self.__training_targets[p : p + 1])
        
        #Calculate Deltas
        l2_delta = layer_3 - self.__training_targets[p : p + 1]
        l1_delta = l2_delta.dot(self.__weights_12.T) * self.RELU2DERIV(layer_2)
        
        #Update the Weights       
        
        self.__weights_12 += self.__lr * layer_2.T.dot(l2_delta)
        self.__weights_01 += self.__lr * layer_1.T.dot(l1_delta)

    def train(self, epoch):
        for i in range(epoch):
            self.__error = 0.0
            self.__correct_count = 0
            
            print '[-] Epoch ', i
            
            for datapoint in range(len(self.__training_dataset)):
                self.backpropagation(datapoint)
            print "Error = ", self.__error
            


