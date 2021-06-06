''' 
=======================================
THREE INPUTS WITH SINGLE NEURON
=======================================
'''
inputs  = [
            1.2 ,5.1 ,2.1
          ]

weights = [
            3.1 ,2.1 ,8.7
          ]

bias    = 3

output = 0
for i in range(3) :
    output += inputs[i]*weights[i]
output += bias
print("[+]Single neuron with three inputs : " ,output)
'''
======================================
FOUR INPUTS WITH SINGLE NEURON
======================================
'''
inputOne = [
                1 ,2 ,3 ,4
           ]

weightOne= [
                1.2 ,4.5 ,6.6 ,-0.3
           ]
bias = 9
outputOne = 0
for i in range(4) :
    outputOne += inputOne[i]*weightOne[i]
outputOne += bias
print("[+]Single neuron with four inputs : " ,outputOne)
'''
======================================
FOUR INPUTS WITH THREE NEURON
======================================
'''
input = [
            1 ,2 ,3 ,4
        ]

weight1 = [
            0.3 ,0.4 ,0.3 ,4.3
          ]
          
weight2 = [
            0.13 ,0.34 ,0.31 ,2.2
          ]

weight3 = [
            0.73 ,0.64 ,0.38 ,2.1
          ]

bias1 = 23
bias2 = 34 
bias3 = 9

output0 = 0
output1 = 0
output2 = 0

for i in range(4) :
    output0 += input[i]*weight1[i]
    output1 += input[i]*weight2[i]
    output2 += input[i]*weight3[i]
output0 += bias1
output1 += bias2
output2 += bias3

Layer = [
            output0 ,output1 ,output2
          ]
print("[+] Layer : " ,Layer)