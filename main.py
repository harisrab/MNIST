import numpy as np
from neural_container import ANN
from keras.datasets import mnist
import sys
from PIL import Image
import os
import random


print '[+] Dataset import completed'

print '[+] All Libraries Imported'

def Process_Data():
    
    'Acquire training dataset'

    (x_train, y_train) = mnist.load_data()[0]

    images, labels = (x_train[0 : 1000].reshape(1000, 28 * 28) / 255, y_train[0 : 1000])
    

    binary_labels = np.zeros((len(labels), 10))

    for i, l in enumerate(labels):
        binary_labels[i][l] = 1

    labels = binary_labels
    
    return images, labels

def main():

    training_data, training_targets = Process_Data()

    print '[+] Training Dataset Acquired'
    
    #Neural Network Setup Parameters
    alpha            = 2
    epochs           = 20
    pixels_per_image = 784
    num_labels       = 10
    batch_size       = 128

    #Instantiate Neural Network
    Img_Recognizer = ANN(training_data, training_targets, alpha, epochs, pixels_per_image, num_labels, batch_size)
    PATH = '/home/skylake/UDataScience/MNIST/assets/hello/'
    c = 2

    while True:
        
        print ''
        print '[+] Select Options for Image Recognizer \n'
        print '1. Train the neural network'
        print '2. Rename the Images in the assets folder'
        print '3. Exit'
        print ''

        option = int(input('> Enter your option: '))
        
        if (option == 1 or c == 0):
            """ Train Neural Network """
            
            print '[+] Initilizing Training Sequence'

            Img_Recognizer.train()
            
            print '[+] Training Complete '

        elif (option == 2):
            """ Tests the neural network """

            while (len(os.listdir(PATH)) != 0):
                filename = random.choice(os.listdir(PATH))
                
                Img= Image.open(PATH + filename)
                inputs = np.asarray(Img)
                inputs = inputs.reshape(1,784) / 255

                number = str(Img_Recognizer.detect_number_present(inputs)) + ".jpg"
                os.rename(os.path.join(PATH, filename), os.path.join(PATH, number))



        elif (option == 3):
            break


    pass



if __name__ == '__main__':
    main()

