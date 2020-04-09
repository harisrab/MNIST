import numpy as np
import sys

from keras.datasets import mnist

class ANN:

    def __init__(self,training_data, training_targets, alpha, epochs, pixels_per_image, num_labels, batch_size):
        """Neural Network's Initializer"""
        
        print '[+] Neural Network Initilized'
        
        #Initilize the parameters
        self.__training_data    = training_data
        self.__training_targets = training_targets
        self.__alpha            = alpha
        self.__epochs           = epochs
        self.__pixels_per_image = pixels_per_image
        self.__num_labels       = num_labels
        self.__batch_size       = batch_size
        
        self.__error       = 0.0
        self.__correct_cnt = 0
       
        
        #Initlize the seed
        np.random.seed(1)
        
        
        
        #Activation Functions
        self.__tanh       = lambda x : np.tanh(x)
        self.__tanh2deriv = lambda x : (1 - (x ** 2))
        self.__softmax    = lambda x : (np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True))

        '''Convolutional Parameters'''

        #Dimensions of Orignal Image
        self.__input_rows       = 28
        self.__input_columns    = 28
    
        #Dimensions of Subsection
        self.__kernel_rows      = 3
        self.__kernel_columns   = 3

        self.__num_kernels      = 16

        self.__hidden_size      = ((self.__input_rows - self.__kernel_rows) * (self.__input_columns - self.__kernel_columns)) * self.__num_kernels 
    
        self.__kernels          = 0.02 * np.random.random((self.__kernel_rows * self.__kernel_columns, self.__num_kernels)) - 0.01
        
        #Initilize Weight Matrices
        self.__weights_12 = 0.2 * np.random.random((self.__hidden_size, self.__num_labels)) - 0.1
            
        print '[+] Neural Parameters Established'
        print ''
        
       
    def train(self):
        for j in range(self.__epochs):
            self.__correct_cnt = 0
            
            for i in range(int(len(self.__training_data) / self.__batch_size)):
                
                batch_start, batch_end = ((i * self.__batch_size),((i + 1) * self.__batch_size))
              
                layer_0 = self.__training_data[batch_start:batch_end]
                layer_0 = layer_0.reshape(layer_0.shape[0],28,28)
                layer_0.shape

                sects = list()
                for row_start in range(layer_0.shape[1]- self.__kernel_rows):
                    for col_start in range(layer_0.shape[2] - self.__kernel_columns):
                        sect = self.get_image_section(layer_0,
                                         row_start,
                                         row_start+ self.__kernel_rows,
                                         col_start,
                                         col_start+ self.__kernel_columns)
                        sects.append(sect)

                expanded_input = np.concatenate(sects,axis=1)
                es = expanded_input.shape
                flattened_input = expanded_input.reshape(es[0]*es[1],-1)

                kernel_output = flattened_input.dot(self.__kernels)
                layer_1 = self.__tanh(kernel_output.reshape(es[0],-1))
                dropout_mask = np.random.randint(2, size = layer_1.shape)
                layer_1 *= dropout_mask * 2
                layer_2 = self.__softmax(np.dot(layer_1,self.__weights_12))
               

                for k in range(self.__batch_size):
                    labelset = self.__training_targets[batch_start + k:batch_start+k+1]
                    _inc = int(np.argmax(layer_2[k:k+1]) == np.argmax(labelset))
                    self.__correct_cnt += _inc

                       
                layer_2_delta = (self.__training_targets[batch_start : batch_end] - layer_2) / (self.__batch_size * layer_2.shape[0])
                layer_1_delta = layer_2_delta.dot(self.__weights_12.T) * self.__tanh2deriv(layer_1)
                layer_1_delta *= dropout_mask
                self.__weights_12 += self.__alpha * layer_1.T.dot(layer_2_delta)
                l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
                k_update = flattened_input.T.dot(l1d_reshape)
                self.__kernels -= self.__alpha * k_update
             
            if(j % 1 == 0):
                sys.stdout.write("\n"+ "Epoch = " + str(j) + " |  Train-Acc = " + str(self.__correct_cnt/float(len(self.__training_data))))
        
    def detect_number_present(self, vector):

        layer_0 = vector
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape
        
        sects = list()

        for row_start in range(layer_0.shape[1] - self.__kernel_rows):
            for col_start in range(layer_0.shape[2] - self.__kernel_columns):
                sect = self.get_image_section(layer_0, row_start, row_start + self.__kernel_rows, col_start, col_start + self.__kernel_columns)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis = 1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(self.__kernels)
        layer_1 = self.__tanh(kernel_output.reshape(es[0], -1))

        
        layer_2 = self.__softmax(np.dot(layer_1, self.__weights_12))

        max_index = np.unravel_index(np.argmax(layer_2, axis=None), layer_2.shape)[1]


        return max_index


    

    def get_image_section(self, layer,row_from, row_to, col_from, col_to):
        section = layer[:,row_from:row_to,col_from:col_to]
        return section.reshape(-1,1,row_to-row_from, col_to-col_from)
              

