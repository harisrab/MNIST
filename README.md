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
### Pre-Processing the Dataset for Testing

We do the same as above with dataset for testing. The only difference is that we will not slice the data.

```python
1  def Preprocess_Test_Data():
2      """Method for Test Data Segregation"""
3      (x_test, y_test) = mnist.load_data()[1]
4      
5      test_images = x_test.reshape(len(x_test),28*28) / 255
6      test_labels = np.zeros((len(y_test),10))
7  
8      for i,l in enumerate(y_test):
9           test_labels[i][l] = 1
10               
11     return test_images, test_labels
```
### Building an Organized, Extensible Framework for Neural Network

The architecture that I have designed here can be further improved and new features can be added to it. This makes it very easy for anyone to understand how the whole process works. I've made a seperate file to contain this, so to make the code less cluttered. 

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

        self.__hidden_size      = ((self.__input_rows - self.__kernel_rows) * (self.__input_columns - self.__kernel_columns)) * self.__num_kernels 
    
        self.__kernels          = 0.02 * np.random.random((self.__kernel_rows * self.__kernel_columns, self.__num_kernels)) - 0.01
        
        #Initilize Weight Matrices
        self.__weights_12 = 0.2 * np.random.random((self.__hidden_size, self.__num_labels)) - 0.1
        #self.__weights_12 = 0.2 * np.random.random((hidden_size,num_labels)) - 0.1
        
        print '[+] Neural Parameters Established'
        print ''

```



You can also:
  - Import and save files from GitHub, Dropbox, Google Drive and One Drive
  - Drag and drop markdown and HTML files into Dillinger
  - Export documents as Markdown, HTML and PDF

Markdown is a lightweight markup language based on the formatting conventions that people naturally use in email.  As [John Gruber] writes on the [Markdown site][df1]

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.

This text you see here is *actually* written in Markdown! To get a feel for Markdown's syntax, type some text into the left window and watch the results in the right.

### Tech

Dillinger uses a number of open source projects to work properly:

* [AngularJS] - HTML enhanced for web apps!
* [Ace Editor] - awesome web-based text editor
* [markdown-it] - Markdown parser done right. Fast and easy to extend.
* [Twitter Bootstrap] - great UI boilerplate for modern web apps
* [node.js] - evented I/O for the backend
* [Express] - fast node.js network app framework [@tjholowaychuk]
* [Gulp] - the streaming build system
* [Breakdance](https://breakdance.github.io/breakdance/) - HTML to Markdown converter
* [jQuery] - duh

And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

### Installation

Dillinger requires [Node.js](https://nodejs.org/) v4+ to run.

Install the dependencies and devDependencies and start the server.

```sh
$ cd dillinger
$ npm install -d
$ node app
```

For production environments...

```sh
$ npm install --production
$ NODE_ENV=production node app
```

### Plugins

Dillinger is currently extended with the following plugins. Instructions on how to use them in your own application are linked below.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |


### Development

Want to contribute? Great!

Dillinger uses Gulp + Webpack for fast developing.
Make a change in your file and instantaneously see your updates!

Open your favorite Terminal and run these commands.

First Tab:
```sh
$ node app
```

Second Tab:
```sh
$ gulp watch
```

(optional) Third:
```sh
$ karma test
```
#### Building for source
For production release:
```sh
$ gulp build --prod
```
Generating pre-built zip archives for distribution:
```sh
$ gulp build dist --prod
```
### Docker
Dillinger is very easy to install and deploy in a Docker container.

By default, the Docker will expose port 8080, so change this within the Dockerfile if necessary. When ready, simply use the Dockerfile to build the image.

```sh
cd dillinger
docker build -t joemccann/dillinger:${package.json.version} .
```
This will create the dillinger image and pull in the necessary dependencies. Be sure to swap out `${package.json.version}` with the actual version of Dillinger.

Once done, run the Docker image and map the port to whatever you wish on your host. In this example, we simply map port 8000 of the host to port 8080 of the Docker (or whatever port was exposed in the Dockerfile):

```sh
docker run -d -p 8000:8080 --restart="always" <youruser>/dillinger:${package.json.version}
```

Verify the deployment by navigating to your server address in your preferred browser.

```sh
127.0.0.1:8000
```

#### Kubernetes + Google Cloud

See [KUBERNETES.md](https://github.com/joemccann/dillinger/blob/master/KUBERNETES.md)


### Todos

 - Write MORE Tests
 - Add Night Mode

License
----

MIT


**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>

