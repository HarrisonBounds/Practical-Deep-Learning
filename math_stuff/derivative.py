import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5

p = f(3.0)
print(p) #we receive 20 here

xs = np.arange(-5, 5, 0.25) #x axis values
print(xs)
ys = f(xs) #y axis values
print(ys)

#plot the graph : notice that the input we put earlier (3) matches with the output of y (20)
plt.plot(xs, ys)
#plt.show()

#What is the derivative of this function at any single point of x?
def derivative(): #As h gets smaller and smaller, how does x respond?
    h = 0.001
    x = 3.0
    print(f(x + h)) #Earlier is was 20, now it is 20.014 because it was slightly nudged in the positive direction
    print((f(x + h) - f(x))) #this number is how much the function responded
    print((f(x + h) - f(x)) / h) #normalize by rise / run to get the slope : 14 in this case

    #we can also get this answer by deriving the return value of f using the product rule


def more_complex():
    h = 0.0001

    #inputs
    a = 2.0
    b = -3.0
    c = 10.0

    d1 = a*b + c #d is a function of 3 scalar inputs
    a += h 
    d2 = a*b + c
    print("d1 with respect to a: ", d1)
    print("d2 with respect to a: ", d2)
    print("slope: ", (d2 - d1) / h) #d2-d1 is how much the function increased whe nwe bumped up a specific input

    #reset a 
    a = 2.0
    d1 = a*b + c 
    b += h #since b is negative, increasing this and multiplying this by a will add more to d
    d2 = a*b + c
    print("d1 with respect to b: ", d1)
    print("d2 with respect to b: ", d2)
    print("slope: ", (d2 - d1) / h) 

    #reset b
    b = -3.0
    d1 = a*b + c 
    c += h #since c is by itself as a scalar and its positive, this will increase d if we add more 
    d2 = a*b + c
    print("d1 with respect to c: ", d1)
    print("d2 with respect to c: ", d2)
    print("slope: ", (d2 - d1) / h) 


#Neural Networks are full of mathematical operations such as these with a huge number of inputs