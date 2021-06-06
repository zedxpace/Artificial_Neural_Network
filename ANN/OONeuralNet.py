##Batches : calculation parallely ,Generalization
import numpy as np 
'''
Inputs for layer one
    - inputs
    - weights
    - biases
'''
inputs = [
         [1 ,2 ,3 ,2               ] ,
         [2.0 ,5.0 ,-1.0 ,2.0       ] ,
         [-0.26 ,-0.27 ,0.17 ,0.87  ] 
         ]

weights = [
          [0.2 ,0.8 ,-0.5 ,1.0      ] ,
          [0.5 ,-0.91 ,0.26 ,-0.5   ] ,
          [-0.26 ,-0.27 ,0.17 ,0.87 ] 
          ]

biases = [2 ,3 ,0.5                 ] 


# inputs1 = [
#           [0.2 ,0.8 ,-0.5 ,1.0       ] ,
#           [0.5 ,-0.91 ,0.26 ,-0.5       ] ,
#           [-0.26 ,-0.27 ,0.17 ,0.87  ] 
#           ]

weights1 = [
           [0.1 ,-0.14 ,0.5          ] ,
           [-0.5 ,0.12 ,-0.33        ] ,
           [-0.44 ,0.73 ,0.13        ] 
           ]

biases1  = [2 ,3 ,0.5                ]
'''
Inputs for layer two will be the output of previous layer
    - outputLayer
    - weights1
    - biases1
'''
outputLayer1 = np.dot(inputs ,np.array(weights).T) + biases
print("##Layer1 = inputs*weights + biases\n" ,outputLayer1)

outputLayer2 = np.dot(outputLayer1 ,weights1) +biases1
print("##Layer2 = outputLayer1*weights1 + biases1\n" ,outputLayer2)

'''
Object Oriented Programming 
X -> Training dataset
##We intitalize the values with random values between the positive one and negative one
'''
print("===============================")
import numpy as np

X = [
        [1 ,2 ,3 ,2.5] ,
        [2.0 ,5.0 ,-1.0 ,2.0] ,
        [-1.5 ,2.7 ,3.3 ,-0.8]
    ]

class LayerDense :
    def __init__(self ,n_inputs ,n_neurons) :
        self.weights = 0.10*np.random.randn(n_inputs ,n_neurons)
        self.biases  = np.zeros((1 ,n_neurons))
    def forward(self ,inputs)  :
        self.LayerOutput = np.dot(inputs ,self.weights) + self.biases

'''
Creating Layer
         ##LayerDense(inputSize ,NumberOfNeurons)
'''
LayerOne = LayerDense(4 ,5)
LayerTwo = LayerDense(5 ,9)

##ForwardPropogration
LayerOne.forward(X)
print("##Layer1\n" ,LayerOne.LayerOutput)

LayerTwo.forward(LayerOne.LayerOutput)
print("##Layer2\n" ,LayerTwo.LayerOutput)