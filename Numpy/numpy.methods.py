import numpy as np
## arange
arange = np.arange(0,20,2)
print("Arange: " + str(arange))

## zeros
zeros = np.zeros((9))
print("Zeros: " + str(zeros))

## linspace
linspace = np.linspace(0,20,6)
print("Linspace: " + str(linspace))

## eye -> create a matrix
eye = np.eye(3)
print("Eye: " + str(eye))

## random
random = np.random.randn(4)
print("Random array: " + str(random))
random = np.random.randint(1,11)
print("Random: " + str(random))
