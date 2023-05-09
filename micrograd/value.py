import math
from graphviz import Digraph
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

    def __add__(self, other):
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
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            #Using the chain rule to compute the gradient in backpropogation
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    #This method is for multiplting a number to an instance of Value when they are reversed (2 * a instead of a * 2) 
    #because Python cannot recognize it 
    def __rmul__(self, other): 
        return self * other
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1) 
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * 2 #Combination of derivative for tanh and chain rule
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


