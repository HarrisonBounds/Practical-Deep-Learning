import numpy as np

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

