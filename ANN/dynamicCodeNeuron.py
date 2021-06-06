input = [
            1 ,2 ,3 ,4
        ]

weights = [
            [0.3 ,0.4 ,0.3 ,4.3],
            [0.13 ,0.34 ,0.31 ,2.2],
            [0.73 ,0.64 ,0.38 ,2.1],
          ]

biases = [
            1 ,2 ,3
         ]

Layer_output = []   #output of current layer

for neuron_weight ,neuron_biases in zip(weights ,biases) :

    neuron_output = 0 #output of given neuron
    for n_input ,weight in zip(input ,neuron_weight) :
        neuron_output += n_input*weight

    neuron_output += neuron_biases
    Layer_output.append(neuron_output)

print("[+] Layer : " ,Layer_output)
'''
DOT Product of vector and matrix ##
'''
import numpy as np
output  =  np.dot(weights[0] ,input) + biases[0]

print("[+] Single neuron : " ,output)
output = np.dot(weights ,input) + biases
print("[+] Three neuron : " ,output)