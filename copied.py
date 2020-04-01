import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = x_train[0 : 1000].reshape(1000, 784) , y_train[0 : 1000]

one_hot_labels = np.zeros((len(y_test), 10))

for i,l in enumerate(labels):
    one_hot_labels[i][l]= 1

np.random.seed(1)

relu = lambda x : (x >= 0) * x
relu2deriv = lambda x : x >= 0

alpha = 0.005
epochs = 350
hidden_size = 40
pixels_per_img = 784
num_labels = 10

weights_01 = 0.2 * np.random.random((pixels_per_img, hidden_size)) - 0.1
weights_12 = 0.2 * np.random.random((hidden_size, num_labels)) - 1

for j in range(epochs):
    error = 0.0
    correct_cnt = 0

    for i in range(len(images)):
        #Forward Propagation
        layer_0 = images[i : i + 1]
        layer_1 = relu(np.dot(layer_0, weights_01))
        layer_2 = np.dot(layer_1, weights_12)

        #Error
        error += np.sum((labels[i : i + 1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == \
                np.argmax(labels[i : i + 1]))

        layer_2_delta = (labels[i : i + 1] - layer_2)
        layer_2_delta = layer_2_delta.dot(weights_12.T) \
                *relu2deriv(layer_1)

        weights_12 += alpha * layer_1.T.dot(layer_2_delta)
        weights_01 += alpha * layer_0.T.dot(layer_1_delta)

    sys.stdout.write("\r" + \
                     "Correct: " + str(correct_cnt/float(len(images))))

