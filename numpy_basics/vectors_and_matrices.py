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

def zeros_and_ones():
    #Create an array that is all zeros
    x = np.zeros((2, 3, 4)) #you can read this as two 3x4 matrices
    print("x: ", x)
    print("Shape of x: ", x.shape)
    print("Data type of x: ", x.dtype)
    
    #Declare an array of 0's with a specific data type
    y = np.zeros((10, 10), dtype="uint32")
    print("y: ", y)
    print("Shape of y: ", y.shape)
    print("Data type of y: ", y.dtype)

    ###################################################################

    #Create an array of all ones
    z = np.ones((3, 3)) #3x3 matrix of all ones
    print("z: ", z)

    #can multiply be a scalar to make each element the same. Example:
    z = 8 * np.ones((3, 3))
    print("z after being multiplied by 8: ", z)

    print("\nz's data type: ", z.dtype)

    #To change this type we will need to change each element's type by creating a copy of the array using the keyword 'astype'
    p = z.astype("uint8")
    print("\np: ", p)
    print("data type of p: ", p.dtype)

    #We can also create copies by using the .copy() method, string splicing z[:], and the built in arange function
    #I like using the copy the best, but they are all useful
    a = z.copy()
    b = z[:]
    
    print("\na: ", a)
    print("\nb: ", b)

def indexing_and_slicing():
    a = np.zeros((3, 4), dtype="uint8")
    print("a: ", a)

    #accessing a matrix is similar to accessing a 2d array. Be sure to specify the row and column first.. and make sure you start at 0!
    a[0, 1] = 1
    a[1, 0] = 2
    a[2, 2] = 3

    print("a after changes: ", a)

    #To access an element, use the standard notation a[][]
    print("The first change: ", a[0][1])

    #To return an entire row, just specify one bracket
    print("Second row of a: ", a[1])

    #######################################################################

    b = np.arange(10) #Will create a vector will elements 0-9
    print("b: ", b)

    #Acesses elements 1-3
    print("Elements 1-3: ", b[1:4])

    #You can include a third argument that specifies the step size. If you do not include it, it is assumed to be one
    print("All even elements: ", b[0:9:2])

    #You can also slice an array up to a certain element or starting from a certain element
    print("b starting from element 5", b[5:])
    print("b up until element 5: ", b[:5])

    #Other neat features
    print("\nLast element in b: ", b[-1])

    print("\nb printed backwards: ", b[::-1])

    #You can also use slicing on matrices
    c = np.arange(20).reshape((4, 5)) #declares a vector and then reshapes it into a matrix
    print("c as a matrix: ", c)

    #The first argument in this row is the row ([1:3] is asking for rows 1 and 2] and the second argument is columns ([:] is all columns)
    print("\nrows 1 and 2 of c: ", c[1:3,:])
    print("\nlast few elements of the last rows: ", c[2:, 2:])


def broadcasting():
    a = np.arange(5)
    c = np.arange(5)[::-1]
    x = np.arange(25).reshape((5, 5))
    y = np.arange(30).reshape((5, 6))

    print("a: ", a)

    print("\na * PI: ", a*3.14)

    print("\na * a: ", a * a)
    
    print("\na * c: ", a * c) #since a and c are the same size, they can be multiplied (rule for vectors)

    #Rule for matrices: To multiply, the number of columns in the first matrix must be equal to the number of rows in the second matrix
    print("\na * x: (a(1x5) x(5x5)): ", a * x)

    #Also used for the dot product
    print("\na dot a: ", np.dot(a, a)) #returns a scalar

    #Dot product for two matrices
    print("\nx dot y: ", np.dot(x, y))
broadcasting()
    
