import math
import random
from typing import Any
from graphviz import Digraph
import torch
#I will be using the github repo "Micrograd" from Andrej Karpathy to illustrate and understand backpropogation

#We need a data structure to maintain the mathematical expressions that make up a Neural Network

class Value:

    def __init__(self, data, _children=(), _op='', label=''): #childern is set to an empty tuple here, but when we maintain it in the class it is a set : _op (operation) is the empty set for leaves
        self.data = data
        self._prev = set(_children)
        self.grad = 0 #The grad as 0 represents no change
        self._backward = lambda: None #backward function doesn't do anything by default
        self._op = _op
        self.label = label

    def __repr__(self): #Wrapper function automatically returns the string
        return f"Value(data={self.data})"
    
    #We need to use the __ naming mehtod to define these operators within the class

    def __add__(self, other): #self + other
        #The expression below reads if other is an instance of Value, we leave it alone
        #but if other is not as instance of value, we will make it an instance of value
        #This way we can simply add integers and wahtnot without explicity having to make them instances of the Value class
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        #This function is for backpropogation
        def _backward(): 
            #These are ways you compute the gradient of each step : If the output is added, then you take the local derivative with respect to the final output... which would be 1
            self.grad += 1.0 * out.grad #we need to accumulate these gradients according to the multivariate case of the chain rule (if we use a variable more than once)
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other): #self - other
        return self + (-other)
    
    def __sub__(self, other): #self - other
        return self + (-other)
    
    def __rsub(self, other):
        return other + (-self)
    
    def __neg__(self): #-self
        return self * -1
    
    def __mul__(self, other): #self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            #Using the chain rule to compute the gradient in backpropogation
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    #This method is for multiplting a number to an instance of Value when they are reversed (2 * a instead of a * 2) 
    #because Python cannot recognize it 
    def __rmul__(self, other): #other * self
        return self * other
    
    def __pow__(self, other): 
        other = other if isinstance(other, (int, float)) else Value(other) #make sure it is a float or an int
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other): #self / other
        return self * other ** -1 #This is another way of doing divison
    
    def __rtruediv__(self, other):
        return other * self ** -1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) 
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * 2 #Combination of derivative for tanh and chain rule
        out._backward = _backward
        return out
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad #derivative of e^x is just e^x which we computed in out.data 

        out._backward = _backward
        return out
    def backward(self):
        
        #To not have to manually call ._backward for each pass, we will use a topological sort
        #This is basically where you lay a graph in a linear fashion so that each node flow from left to right
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

            for child in v._prev:
                build_topo(child)
            topo.append(v)

        build_topo(self)

        self.grad = 1.0 #set the gradient to 1.0 (since by default it is 0)

        for node in reversed(topo):
            node._backward()
def manual_define():   
    #inputs x1, x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    #weights w1, w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    #bias of the neuron
    b = Value(6.881373587019543, label='b')

    #manually building the chain
    x1w1 = x1*w1; x1w1.label = "x1*w1"
    x2w2 = x2*w2; x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

    n = x1w1x2w2 + b; n.label = 'n' #neuron
    o = n.tanh(); o.label = 'o' #output

def pytorch_define():
    x1 = torch.Tensor([2.0]).double() ; x1.requires_grad = True #We are casting these tensors to doubles becasue the default type for a tensor is float32 
    x2 = torch.Tensor([0.0]).double() ; x2.requires_grad = True #WE haveto specify that these all require gradients, because they are all leaf nodes meaning gradients are set to False by default
    w1 = torch.Tensor([-3.0]).double() ; w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double() ; w2.requires_grad = True
    b = torch.Tensor([6.881373587019543]).double() ; b.requires_grad = True
    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)

    print(o.data.item()) #item returns the elements in the tensor and not the tensor itself
    o.backward()

    print('-----')
    print('x2', x2.grad.item())
    print('w2', w2.grad.item())
    print('x1', x1.grad.item())
    print('w1', w1.grad.item())


def example():
    #For example
    print(Value(2.0))

    #When we implement the add method, Python will call it like so: a.__add__(b)
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10)

    d = a*b + c

    print(d) #(a.__mul__(b)).__add__(c)

    #Print the children of d
    print(d._prev) #We know d was produced by what this returns

    print(d._op) #We know this is the operator that was used to produce d

def example2():
    a = Value(2.0)
    b = Value(4.0)
    print(a - b)
    print(a.exp())
    print(a / b)


class Neuron:
    #The constructor takes in the number of inputs coming into the Neuron and randomly assigns them a weight and a bias
    #Weight is influence, bias is overall trigger happiness of the neuron
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

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
    
#All forward pass examples
def neuronExample():
    x = [2.0, 3.0]
    n = Neuron(2)
    print(n(x)) #This is the forward pass for a single neuron

def layerExample():
    x = [2.0, 3.0]
    n = Layer(2, 3) #2 inputs, 3 outputs
    print(n(x))

def MLPExample():
    x = [2.0, 3.0, -1.0] #3 inputs
    n = MLP(3, [4, 4, 1]) #3 inputs into two layers of 4 and 1 output
    print(n(x))



def loss():
    
    n = MLP(3, [4, 4, 1]) #3 inputs into two layers of 4 and 1 output
    
    #Multiple input datas                    
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0] #This is the desired output for each of the inputs
    ypred = [n(x) for x in xs]

    

    #ypred = [x[0].data for x in ypred] #extracting the values from the ypred list

    print('ys: ', ys)
    print('ypred: ', ypred)

    #Mean Squared Error Loss
    #For each of the inputs, we are subtracting the prediction(ypred) and the groundtruth(ys) and then squaring them
    #This is a way to tell how far off your prediction is to the desired target
    #You only get 0 when the ground truth and prediction are equal to each other
    #The more off we are from the target, the greater the loss
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)) #This is the total loss
    print(loss)

    #Now we need to minimize the loss so that each of the predicitons is close to its target

    loss.backward()

    print("Before\n-----")
    print(n.layers[0].neurons[0].w[0].grad)
    print(n.layers[0].neurons[0].w[0].data)

loss()    
def parametersExample():
    x = [2.0, 3.0, -1.0] #3 inputs
    n = MLP(3, [4, 4, 1])
    print(n.parameters()) #All of the weights and biases inside of the neural network
    print(len(n.parameters()))


    x = [2.0, 3.0, -1.0] #3 inputs
    n = MLP(3, [4, 4, 1])

    #look at the data before we do gradient descent
    print("Before\n-----")
    print(n.layers[0].neurons[0].w[0].grad)
    print(n.layers[0].neurons[0].w[0].data)

    for p in n.parameters():
        #Think of the gradients as vectors that point in the direction to increase the loss
        #THis is why we need a negative sign in the stepping scalar, because we want to decrease the loss
        p.data += -0.01 * p.grad #minimizing the loss 

    #look at the data after gradient descent
    print("After\n-----")
    print(n.layers[0].neurons[0].w[0].grad)
    print(n.layers[0].neurons[0].w[0].data)


