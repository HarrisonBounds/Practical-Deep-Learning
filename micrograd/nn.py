import driver
import random

class Neuron:
    #The constructor takes in the number of inputs coming into the Neuron and randomly assigns them a weight and a bias
    #Weight is influence, bias is overall trigger happiness of the neuron
    def __init__(self, nin):
        self.w = [driver.Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = driver.Value(random.uniform(-1, 1))

    def __call__(self, x):
        #w * x + b : where w * x is the dot product
        #notation is n(x)
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) + self.b #zip will iterate over the tuples of the parameteres : This means that it will pair up the w and x 
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

#A Layer is just a list of neurons 
class Layer:
    def __init__(self, nin, nout): #How many neurons do you want in your layer = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            np = neuron.parameters() #parameters of the neuron
            params.extend(np)
        return params

class MLP: #Multi Layered Perceptron

    def __init__(self, nin, nouts): #nouts is now a list, meaning it is the sizes of all of the layers in our MLP
        sz = [nin] + nouts #Putting all of the layers together
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] #Iterating over consecutive pairs of sizes and create layer objects for them

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            lp = layer.parameters() #parameters of the neuron
            params.extend(lp)
        return params