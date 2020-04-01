""" Main Program """

from ANN import neural_network as ANN
import numpy as np
import sys
from keras.datasets import mnist

def main():
    #Load the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #Reshape the Dataset
    images, labels = (x_train[0 : 1000].reshape(1000, 28 * 28) / 255, y_train[0 : 1000])
    
    #Create Empty Matrix 1000 x 10
    one_hot_labels = np.zeros((len(labels), 10))
    
    #Create Output Vectors for each corresponding Image
    for i, l, in enumerate(labels):
        one_hot_labels[i][l] = 1
    labels = one_hot_labels
    

    training_dataset = images
    training_targets = labels
    

    #Instantiate the Neural Recognizer
    Img_Recognizer = ANN(training_dataset, training_targets, 0.005, 40)
    
    epochs = 350

    Img_Recognizer.train(epochs)

    #Img_Recognizer.forward_propagate(test)
    
    #Reshape the Test Data
    test_images = x_test.reshape((len(x_test), 28 * 28))
    test_labels = np.zeros((len(y_test), 10))

    #Populate the test targets
    for i, l in enumerate(y_test):
        test_labels[i][l] = 1

    
        

if __name__ == '__main__':
    main()
