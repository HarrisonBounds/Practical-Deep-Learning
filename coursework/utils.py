import matplotlib.pyplot as plt
import numpy as np
# import sklearn
import sklearn.datasets
# import sklearn.linear_model

def plot_decision_boundary(model, X, y):
    """
    Plot decision boundary on scattered data

    Arguments:
        x -- A scalar or numpy array of any size.

    Return:
    """
    
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    
    s = 1/(1+np.exp(-x))
    
    return s

def load_planar_dataset():
    """
    Generate 2D binary-class data
    """

    np.random.seed(1)
    M = 400 # number of examples
    # N = int(M/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((M,D)) # data matrix where each row is a single example
    y = np.zeros((M,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(int(M/2)*j, int(M/2)*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, int(M/2)) + np.random.randn(int(M/2))*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(int(M/2))*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
        
    # X = X.T
    # Y = Y.T

    return X, y

def load_extra_datasets(): 
    """
    Generate 2D binary classification data using sklearn
    """ 
    M = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=M, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=M, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=M, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=M, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(M, 2), np.random.rand(M, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure