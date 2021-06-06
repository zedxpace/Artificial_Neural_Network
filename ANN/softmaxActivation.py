'''
SOFTMAX ACTIVATION FUNCTION : 
    INPUT -> EXPONENTIATE -> NORMALIZE -> OUTPUT
           ##Combination of exponentiate and normalize is known as softmax  activation function : exponentiate / normalize
First  way to train network is to determine how wrong that network is.
We need to compare neuron to see how accurate they are but ReLU lacks in this.

using exponential function 
     x
y = e
where e(boilers number) = 2.718281828459045
it makes sure no vaue will be negative
'''
import math
import numpy as np
layer_outputs = [4.8 ,1.21 ,2.385]

E = 2.718281828459045   #OR it can be done with math library using E = math.e 

expected_Values = []

for output in layer_outputs :           #OR can be done using expexted_values = np.exp(layer_outputs)
    expected_Values.append(E**output)

print("##Exponential values\n" ,expected_Values)

'''
once we expontiated values next step will be normalizing values
##Normalization
'''
norm_base   = sum(expected_Values)
norm_values =  []

for value in expected_Values :          #OR can be done as norm_values = expected_values / np.sum(expected_values)
    norm_values.append(value / norm_base)

print("##Normalized values\n" ,norm_values)
print("##Sum of normalized values\n" ,sum(norm_values))


##creating neurons batch and applying the softmax function

layer_outputs = [[4.8 ,1.21 ,2.385] ,
                  [8.9 ,-1.81 ,-.2] ,
                  [1.41 ,1.051,0.026] 
                ]
exp_values = np.exp(layer_outputs)
print("## Exponentiated value\n" ,exp_values)

norm_values = exp_values / np.sum(layer_outputs ,axis = 1 ,keepdims=True)
print("## Normalized values\n" ,norm_values)


