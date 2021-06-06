'''
##Step Activation function :
    - Idea is if input is greater than 0 then output
      wil be 1 else output will be 0
    - Activation function comes into  play after the
     'input * weight + bias' operation is done
##Sigmoid Activation function :
    - As we proceed with the weights/biases value ,it will 
      help us to predict the loss or how wrong
      neural network went wrong
    - More Granular output as compared to the Step
      Activation function
##Rectified Linear Activation function :
    - if input is greater than 0 then output is whatever x
      is ,else value is 0.
    - Simple calculation
'''
import numpy as np
import dataSet

##Creating dataSet
dataSet.create_data(100  ,3)

np.random.seed(0)

X = [[1 ,2 ,3 ,2.5],
     [2.0 ,5.0 ,-1.0 ,2.0],
     [-1.5 ,2.7 ,3.3 ,-0.8]]

inputs = [0 ,2 ,-1 ,3.3 ,-2.7 ,1.1 ,2.2 ,-100]

output = []

for i in inputs :
    ##Rectified Linear Activation function
    if i > 0 :
        output.append(i)
    else :
        output.append(0)
    '''
    OR we can simply DO 
    #output.append(max(0 ,i))
    '''

##Layers
class Layer_Dense :
    ##initializing inputs
    def __init__(self ,n_inputs ,n_neurons) :
        self.weights = 0.10 * np.random.randn(n_inputs ,n_neurons)
        ##In case Network is diying by changing all the values to the 0 ,the fix is we can initialize bias with some non-zero number
        self.biases  = np.zeros((1 ,n_neurons))
    
    def forward(self ,inputs) :
        self.output = np.dot(inputs ,self.weights) + self.biases

##Activation function ReLU
class Activation_ReLU :
    def forward(self ,inputs) :
        self.output = np.maximum(0 ,inputs)

##Activation function Softmax
class Activation_softmax :
    def forward(self ,inputs) :
        self.exp_values = (np.exp(inputs) - np.max(inputs ,axis = 1 ,keepdims=True) )
        self.output = self.exp_values / np.sum(self.exp_values ,axis = 1 ,keepdims=True)

##Creating Layers
          ##Layer_Dense(inputs ,neurons)
LAYER = Layer_Dense(4 ,5)
##Object of Activation Function
ACTIVATION = Activation_ReLU() 

##Performing input*weight +bias
LAYER.forward(X)
print("##LAYER-1\n" ,LAYER.output)

##passing output of above call i.e LAYER to the activation function
ACTIVATION.forward(LAYER.output)
print("##LAYER-1 After Activation \n" ,ACTIVATION.output)   #All it did is removed all the negative values from data set passed to it

####
##Testing softmax activation function
'''
1. Creating datset
2. Creating an object of ReLU activation
3. Creating Layer
4. Creating another Layer
5. Creating an object sofmax activation
6. Forwarding Layer X inputs thought network
7. Applying activation function ReLU on the output  of [6]
8. Forwarding output of [7] through the network again 
9. Applying activation function softmax on the output of [7]  
'''
X ,y = dataSet.create_data(100 ,3)                  ##1
activation_ReLU = Activation_ReLU()                 ##2

Layer_Densed = Layer_Dense(2 ,3)                    ##3
Layer_Densed2 = Layer_Dense(3 ,3)                   ##4

activation_softmax = Activation_softmax()           ##5

Layer_Densed.forward(X)                             ##6
activation_ReLU.forward(Layer_Densed.output)        ##7

Layer_Densed2.forward(activation_ReLU.output)       ##8
activation_softmax.forward(Layer_Densed2.output)    ##9

print("##Softmax Activation\n" ,activation_softmax.output)

##How wrong neural network went wrong can be acheved using loss function 
