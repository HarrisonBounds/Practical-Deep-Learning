import driver
import nn
import torch

def manual_define():   
    #inputs x1, x2
    x1 = driver.Value(2.0, label='x1')
    x2 = driver.Value(0.0, label='x2')

    #weights w1, w2
    w1 = driver.Value(-3.0, label='w1')
    w2 = driver.Value(1.0, label='w2')

    #bias of the neuron
    b = driver.Value(6.881373587019543, label='b')

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
    print(driver.Value(2.0))

    #When we implement the add method, Python will call it like so: a.__add__(b)
    a = driver.Value(2.0)
    b = driver.Value(-3.0)
    c = driver.Value(10)

    d = a*b + c

    print(d) #(a.__mul__(b)).__add__(c)

    #Print the children of d
    print(d._prev) #We know d was produced by what this returns

    print(d._op) #We know this is the operator that was used to produce d

def example2():
    a = driver.Value(2.0)
    b = driver.Value(4.0)
    print(a - b)
    print(a.exp())
    print(a / b)

    
#All forward pass examples
def neuronExample():
    x = [2.0, 3.0]
    n = nn.Neuron(2)
    print(n(x)) #This is the forward pass for a single neuron

def layerExample():
    x = [2.0, 3.0]
    n = nn.Layer(2, 3) #2 inputs, 3 outputs
    print(n(x))

def MLPExample():
    x = [2.0, 3.0, -1.0] #3 inputs
    n = nn.MLP(3, [4, 4, 1]) #3 inputs into two layers of 4 and 1 output
    print(n(x))


def lossExample():
    x = [2.0, 3.0, -1.0]
    n = nn.MLP(3, [4, 4, 1]) #3 inputs into two layers of 4 and 1 output
    n(x)
    
    #Multiple input datas                    
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0] #This is the desired output for each of the inputs
    ypred = [n(x) for x in xs]

    print('ypred: ', ypred)

    #Mean Squared Error Loss
    #For each of the inputs, we are subtracting the prediction(ypred) and the groundtruth(ys) and then squaring them
    #This is a way to tell how far off your prediction is to the desired target
    #You only get 0 when the ground truth and prediction are equal to each other
    #The more off we are from the target, the greater the loss
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)) #This is the total loss
    print('loss: ', loss)

    #Now we need to minimize the loss so that each of the predicitons is close to its target

    loss.backward()

    print(n.parameters()) #All of the weights and biases inside of the neural network
    print(len(n.parameters())) #Number of parameters in the neural network


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
    print(n.layers[0].neurons[0].w[0].data)
def train():
    x = [2.0, 3.0, -1.0]
    n = nn.MLP(3, [4, 4, 1]) 
    n(x)

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    #Gradient descent
    #Breaking it up into steps :: This shows the predictions getting closer and closer to their target values

    for k in range(20):
        #forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        #backward pass
        for p in n.parameters():
            p.grad = 0.0 #Set the grads to zero so they dont accumulate 
        loss.backward()

        #update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        print(k, loss.data)
    print(ypred)


