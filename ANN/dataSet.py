import numpy as np

np.random.seed(0)
'''
We are only creating X & Y coordinate
##create_data(featureSet ,classes)
'''
def create_data(points ,classes) :
    X = np.zeros((points * classes ,2))
    Y = np.zeros(points * classes ,dtype = 'uint8')

    for class_number in range(classes)  :
        ix = range(points * class_number ,points * (class_number + 1))
        r  = np.linspace(0.0 ,1 ,points)
        t  = np.linspace(class_number * 4 ,(class_number + 1) * 4 ,points ) + np.random.randn(points) * 0.2
        
        X[ix] = np.c_[r * np.sin(t * 2.5) ,r * np.cos(t * 2.5)]
        Y[ix] = class_number

        return X ,Y
'''
##Can Test Above dataSet 
import matplotlib.pyplot as plt 

print("[+] I am ON")
X ,y = create_data(100 ,3)

plt.scatter(X[:,0] ,X[:,1])
plt.show()

plt.scater(X[: ,1] ,X[: ,1] ,c = y ,cmap = "brg")
plt.show()
'''