import numpy as np

def files():
    a = np.loadtxt("sample_files/abc.txt")
    print("abc.txt: ", a)

    b = np.loadtxt("sample_files/abc_tab.txt")
    print("\nabc_tab.txt: ", b)

    c = np.loadtxt("sample_files/abc.csv", delimiter=",") #The delimeter is what the data is separated by. Usually in csv files it is a comma
    print("\nabc.csv: ", c)

    #How do we write arrays to the disk? Use the .save method and the .npy(numpy) extension to let the disk know it contains a numpy array
    np.save("generated_files/abc.npy", a)

    #to load the array back into memory from disk, we can use the .load method. Note: we have to assign a variable name
    x = np.load("generated_files/abc.npy")
    print("\nx: ", x)

    #To write arrays to files (so that they can be human readable), we can use the .savetxt method
    np.savetxt("generated_files/ABC.txt", b)
    np.savetxt("generated_files/ABC.csv", c)

def multiple_files():
    #Using the savez method, we can write multiple arrays to a disk
    #We can also read these arrays back into memory with load
    a = np.arange(10)
    b = np.arange(20)

    np.savez("generated_files/arrays.npz", a=a, b=b) #use the savez command, save to a .npz file, and set each array equal to its varaible name

    q = np.load("generated_files/arrays.npz") #q is now a dictionary, and the keys are the variable names!

    print("Q keys: ", list(q.keys())) #must cast the keys to a list

    #To access the specific list, we can use this notation q['a'] or q['b']
    print("\nArray a:", q['a'])




