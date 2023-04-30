from PIL import Image
from sklearn.datasets import load_sample_images

#Load the data from the dataset - these are numpy arrays
china = load_sample_images().images[0]
flower = load_sample_images().images[1]

print("China shape and data type: ", china.shape, china.dtype)
print("\FLower shape and data type: ", flower.shape, flower.dtype)

#Since these images are 3 dimensional, they have RGB values : if the images were grayscale, they would only have two dimensions

#Convert the numpy arrays into images
imChina = Image.fromarray(china)
imFlower = Image.fromarray(flower)

#Show the images
imChina.show()

#Save the image
imChina.save("generated_files/china.png")

#Open and show the saved file
im = Image.open("generated_files/china.png")
im.show()


