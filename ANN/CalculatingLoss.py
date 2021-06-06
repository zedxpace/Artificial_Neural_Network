'''
## Calculating Loss with Categorical Cross-Entropy
    - Before implementing BackPropagation ,we need some sort of Matrix which calculates the Loss.
    - But if we focused on increasing the accuracy of the network we will get stuck in binary output only.

## One-hot encoding
    - classes   : n
    - Label     : 0         #this value is basically the index of 1 in below list
    - One-hot   : [1 ,0]   

    - when we define log without its base thenwe are reffering to natural log i.e (ln) 
                    y = log (base e) x = ln(x) i.e euler's number where e is 2.7 blah blah
    
## solvng for x
    e ** x = b where input will be 'b'

import numpy as np
import math

b = 5.2

print(np.log(b))
print(math.e ** 1.6486586255873816)
'''

## Let's get hands dirty
import math

softmax_output  =   [0.7 ,0.1 ,0.2]     #output needs to get from output layer
target_output   =   [1 ,0 ,0]

target_class    =   0                   #Index 0

loss = -(math.log(softmax_output[0]) * target_output[0] +  
        math.log(softmax_output[1]) * target_output[1] +
        math.log(softmax_output[2]) * target_output[2] )

print(loss)
