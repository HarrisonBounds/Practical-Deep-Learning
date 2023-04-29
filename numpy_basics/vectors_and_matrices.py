import numpy as np

#Look at the features of a numpy array
def features():
    
    a = np.array([1,2,3,4])

    print("a:", a)

    print("Size of a: ", a.size)

    print("Shape of a: ", a.shape)

    print("Data type of a: ", a.dtype)

##############################################

#Explicitly declare the data type of the array 

def declare_type():
    b = np.array([1, 2, 3, 4], dtype="uint8")
    print(b.dtype)

    c = np.array([1, 2, 3, 4], dtype="float64")
    print(c.dtype)

##############################################

#Create a matrix

def matrix():
    d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) #note the extra set of brackets
    #Get the shape  of the array (How many rows and columns)
    print("Shape of matrix d", d.shape)

    print("Size of d", d.size) #How many element are in the array

    print("D: ", d) #The array itself (print the matrix)

    #Three dimensional matrix
    e = np.array([[[55, 66, 77], [89, 98, 23]], [[12, 13, 14], [15, 16, 17]]])

    #note the shape in 3d
    print("3d matrix shape: ", e.shape)

matrix()

