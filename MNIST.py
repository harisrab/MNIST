#Import Libraries
import sys
import numpy as np

print '[+] System Libraries Imported'

#Import Dataset
from keras.datasets import mnist

print '[+] Training Libraries Imported'




def Preprocess_Training_Data():
    """Clean and Manage the training data in correct format for input into the neural network"""
    
    (x_train, y_train) = mnist.load_data()[0]

    images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, y_train[0:1000])

    one_hot_labels = np.zeros((len(labels),10))
    for i,l in enumerate(labels):
        one_hot_labels[i][l] = 1
    labels = one_hot_labels
    
    print '[+] Datasets Processed for Input'
    
    return images, labels

def Preprocess_Test_Data():
    """Process the data for test the results on unseen data"""
    
    (x_test, y_test) = mnist.load_data()[1]
    
    test_images = x_test.reshape(len(x_test), 28*28) / 255
    test_labels = np.zeros((len(y_test), 10))
    
    for i, l in enumerate(y_test):
        test_labels[i][l] = 1
    
    return test_images, test_labels
        
class ANN():
    
    def __init__(self,training_data, training_targets, test_data, test_targets, alpha, iterations, hidden_size, pixels_per_image, num_labels, batch_size):
        """Neural Network's Initializer"""
        
        print '[+] Neural Network Initilized'
        
        #Initilize the parameters
        self.__training_data    = training_data
        self.__training_targets = training_targets
        self.__test_data        = test_data
        self.__test_targets     = test_targets
        self.__alpha            = alpha
        self.__epochs           = iterations
        self.__hidden_size      = hidden_size
        self.__pixels_per_image = pixels_per_image
        self.__num_labels       = num_labels
        self.__batch_size       = batch_size
        self.__test_correct_cnt = 0 
        
        self.__error       = 0.0
        self.__correct_cnt = 0
        
        #Initlize the seed
        np.random.seed(1)
        
        #Initilize Weight Matrices
        self.__weights_01 = 0.02 * np.random.random((pixels_per_image,hidden_size)) - 0.01
        self.__weights_12 = 0.2 * np.random.random((hidden_size,num_labels)) - 0.1
        
        #Activation Functions
        self.__relu       = lambda x : (x >= 0) * x 
        self.__relu2deriv = lambda x : (x >= 0)
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

    	self.__hidden_size      = ((input_rows - kernel_rows) * (inputs_cols - kernel_cols)) *num_kernels 
    
    	self.__kernels          = 0.02 * np.random.random.((kernel_rows * kernel_cols, num_kernels)) - 0.01
        
        print '[+] Neural Parameters Established'
        print ''
        
        
        
    def forward_propagation(self, vector):
        """Forward Propagate the Vector and Acquire the Output"""
        
        layer_0 = vector
	layer_0 = layer_0.reshape(layer_0.reshape(layer_0.shape[0], 28, 28))
	layer_0.shape

	sects = list()

	for self.__row_start in range(layer_0.shape[1] - self.__kernel_rows):
		for self.__col_start in range(layer_0.shape[2] - self.__kernel_cols):
			sect = Conv_Subsection(layer_0, self.__row_start, self.__row_start + self.__kernel_rows, self.__col_start, self.__col_start + self.__kernel_cols)
		sects.append(sect)

	expanded_input = np.concatenate(sects, axis = 1)
	es = expanded_input.shape
	flattened_input = expanded_input.reshape(es[0] * es[1], -1)
	
	kernel_output = flattened_input.dot(self.__kernels)
	
        layer_1 = self.__tanh(kernel_output.reshape(es[0], -1))
        layer_2 = np.dot(layer_1, self.__weights_12)
        
	
        return layer_0, layer_1, layer_2

    
    def backpropagation(self, i):
        """Conduct a batchwise backpropagtion to evaluate weights and kernels"""
        
        batch_start, batch_end = ((i * self.__batch_size), ((i + 1) * self.__batch_size))
        
        layer_0 = self.__training_data[batch_start : batch_end]        
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        sects = list()
        for self.__row_start in range(layer_0.shape[2] - self.__kernel_cols):
            for self.__col_start in range(layer_0.shape[0] - self.__kernel_cols):
                sect = self.Conv_Subsection(layer_0, self.__row_start, self.__row_start + self.__kernel_rows, \
					    self.__col_start, self.__col_start + self.__kernel_cols)
                
		sects.append(sects)

        expanded_input = np.concatenate(sects, axis = 1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(self.__kernels)


        layer_1 = self.__tanh(kernel_output.reshape(es[0], -1))
        
        #Add a dropout to the layer
        dropout_mask = np.random.randint(2, size = layer_1.shape)
        layer_1     *= dropout_mask * 2

        layer_2      = self.__softmax(np.dot(layer_1, self.__weights_12))
        
        
        for k in range(self.__batch_size):
            self.__correct_cnt += int(np.argmax(layer_2[k : k + 1]) == np.argmax(self.__training_targets[batch_start + k : batch_start + k + 1]))
                
        layer_2_delta  = (self.__training_targets[batch_start : batch_end] - layer_2) / (self.__batch_size * layer_2.shape[0])
        layer_1_delta  = layer_2_delta.dot(self.__weights_12.T) * self.__tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask
        
        self.__weights_12 += self.__alpha * layer_1.T.dot(layer_2_delta)
        
        l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(l1d_reshape)

        self.__kernels -= self.__alpha * k_update

    def train(self):
        """Batchwise Train the Neural Network"""
        
        for j in range(self.__epochs):
            #self.__error, self.__correct_cnt = (0.0, 0)
            self.__correct_cnt = 0
            
            for i in range(int(len(self.__training_data) / self.__batch_size)):
                
                self.backpropagation(i)
                
                
            self.__test_correct_cnt = 0
            for i in range(len(self.__test_data)):
                layer_0, layer_1, layer_2 = self.forward_propagation(self.__test_data[i : i + 1])
                
                
                self.__test_correct_cnt += int(np.argmax(layer_2) == np.argmax(self.__test_targets[i : i + 1]))

            if(j % 10 == 0):
                
                sys.stdout.write("\n"+ \
                                 "I:" + str(j) + \
                                 " Test-Acc:"+str(self.__test_correct_cnt/float(len(self.__test_data)))+\
                                 " Train-Acc:" + str(self.__correct_cnt/float(len(self.__training_data))))
    

    def Conv_Subsection(self, layer, row_from, row_to, column_from, column_to):
        
        #Cut the piece from the image
        section = layer[:, row_from : row_to, column_from : column_to]
        
        #Reshape the cut out from the image
        section = section.reshape(-1, 1, row_to - row_from, column_to - column_from)
        
        return section


def main(): 
    """Main Function to Understand the Process in Condensed Fashion"""
    
    #Acquire and Clean Test and Training Data
    training_data, training_targets = Preprocess_Training_Data()
    test_images,   test_labels      = Preprocess_Test_Data()
        
    alpha            = 2
    epochs           = 300
    pixels_per_image = 784
    num_labels       = 10
    batch_size       = 100
    
    #Instantiate the Neural Network
    Img_Recognizer = ANN(training_data, training_targets, test_images, test_labels, alpha, epochs, hidden_size, pixels_per_image, num_labels, batch_size)
    
    #Train the Neural Network
    Img_Recognizer.train()


        
if __name__ == '__main__':
    main()
    
    
