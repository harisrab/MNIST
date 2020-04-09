# Barebones Neural Network to Detect Numbers
![Cover](https://deeplizard.com/images/ai-cyborg-cropped-2.png)

Now this was time for me to learn more about the neural networks and discover wonderful mechanisms under the hood of state of the art libraries already available on the market. What could be better than notching up the difficulty level from simulating logic gates to training the brain to detect images from the MNIST Dataset. Above is the code and below is explaination, discussing how does it actually go about detecting images. 

  
## How to Get the Code Up and Running

### Download the Dataset
There are many ways to download dataset, but here is the one that directly imports the dataset into python with 2 lines of code. It's always a good practice to write readable code; therefore, practicing it, is worth it.

Here we install all the necessary back-end libraries that support keras, a deep learning framework, so to access the dataset. Make sure to download pip, which is handy tool to install relevent libraries directly to python.

```sh
$ sudo apt-get update && upgrade
$ sudo apt install python-pip
$ pip install numpy scipy matplotlib scikit-image scikit-learn ipython
$ pip install tensorflow
$ pip install keras
```
Then, in the code, we import the data using the command below. We'll later wash and clean the data so that it's ready to be fed to our neural network.

```python
from keras.datasets import mnist
```
## Explaination for the Moving Parts

### Pre-Processing the Dataset for Training

If you have some experience handling python statements, which I hope you do, then, the code is self-explainatory. For the sake of wholeness I will explain in general terms the working of this code.

Line 4 downloads dataset seperating it into x_train (list of pixel values for each image) and y_train (list of corresponding labels identifying number contained in those images). The we reshape the data by picking up only the first 1000 images and giving a vector form to contain 784 pixel values, which can be fed directly to the neural network. Correspondingly, in line 7, we pick first 1000 labels for reshaped images. 
Here is the most valuable part: using numpy we create an matrix of zeros. Each row in the matrix represents a number, and then for loop is used to iterate through each array and set the corresponding bit of the number to 1. Effectively, this is the representation for the numbers in binary. This format is essential for our neural network.

```python
1 def Preprocess_Training_Data():
2     """Method Segregates Dataset for Training"""
3     
4     x_train, y_train = mnist.load_data()[0]
5 
6     images = x_train[0:1000].reshape(1000,28*28) / 255                  
7     labels = y_train[0:1000]
8 
9
10    one_hot_labels = np.zeros((len(labels),10))
11              
12    for i,l in enumerate(labels):
13        one_hot_labels[i][l] = 1
14    labels = one_hot_labels
15               
16    return images, labels
```

### Building an Organized, Extensible Framework for Neural Network

The architecture that I have designed here can be further improved and new features can be added to it. This makes it very easy for anyone to understand how the whole process works. I've made a seperate file (neural_container.py) to contain this, so to make the code less cluttered. 

```python
class neural_network:
  
  def __init__(self):
    """ Initilizes the variables and weight matrices required for the neural network """
    
    pass
   
  def backpropagation(self):
    """ backpropagates both for normal layers and convolutional layer """
    
    pass
    
  def forward_convolution(self):
    """ Brings forward convolution into play """
    
    pass
    
  def train(self):
    """ Trains the neural network """
    
    pass
    
  def Conv_SubSection(self):
    """ Takes a sub-section snapshot for the image for convolution"""
    
    pass

```

### Populating the constructor for the Neural Network
Code below may seem like a lot, but it really isn't. First we define all the variables that define the architecture of our neural network, some variables contain datasets, both for training and testing. We then use lambda  functions to concisely write activation functions. Rest defines parameters for extraction of sub-section of the image and dimensions of the convolutional layer. At last we define weights for the convolutional kernel and simple weight matrices between hidden and output layer of the neural network. 

```python

def __init__(self,training_data, training_targets, test_data, test_targets, alpha, iterations, pixels_per_image, num_labels, batch_size):
        """Neural Network's Initializer"""
        
        print '[+] Neural Network Initilized'
        
        #Initilize the parameters
        self.__training_data    = training_data
        self.__training_targets = training_targets
        self.__test_data        = test_data
        self.__test_targets     = test_targets
        self.__alpha            = alpha
        self.__epochs           = iterations
        self.__pixels_per_image = pixels_per_image
        self.__num_labels       = num_labels
        self.__batch_size       = batch_size
        self.__test_correct_cnt = 0 
        
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
        #self.__weights_12 = 0.2 * np.random.random((hidden_size,num_labels)) - 0.1
        
        print '[+] Neural Parameters Established'
        print ''

```

### Populate the Forward Propagation

We've named this function, which is a part of the above class, detect_number_present. This takes in a vectorized image of a number from MNIST test dataset, of shape (1, 784), to throughput a number that this neural network thinks is contained within the image. This network contains one convolutional layer that convolves the image from left to right to extract features such as curves, edges, which maybe unique to each of the numbers. This helps distinguish between numbers 2 and 3 which may, otherwise, seem simillar. The network must be trained before initiating forward propagation, which, otherwise, would produce dubious results.

```python
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
        
```
### Main Function 

In the main function we summarize the functionality of the neural network. We call the function to process the dataset for training, segregating it into labels and inputs. Then, comes the defination  of parameters for neural container, which are determined by trial and error for best performance. 

Then we have a very basic front-end for user interaction that displays options to train, and subsequently, use the neural network with trained weights to predict numbers contained in images. Program iterates through every file in the assets folder, vectorizes it, and feeds it through the neural network to rename the image with the detected number. 

![](https://github.com/harisrab/MNIST/blob/master/resource/Screenshot%20from%202020-04-08%2021-07-22.png)

![](https://github.com/harisrab/MNIST/blob/master/resource/Screenshot%20from%202020-04-08%2021-12-53.png)



