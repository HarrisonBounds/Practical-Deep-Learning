#Bigram language model - statistical lanugage model that predicts the probabilty of a character/word based on a previous character/word
#Bigram - A piar of consecutive characters/words

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('makemore/names.txt', 'r').read().splitlines() #Puts each name in the text file into a list of words

N = torch.zeros((27, 27), dtype=torch.int32) #Using PyTorch to set a 27x27 (26 letter of the alphabet + start and end token) array with the type of a 32 bit integer

chars = sorted(list(set(''.join(words)))) #storing each character from the alphabet in a list
stoi = {s:i+1 for i, s in enumerate(chars)} #mapping each character to an integer index : offset by 1 to have the '.' as index 0
stoi['.'] = 0 #Instead of having two tokens, we will just use one (Make this position 0 and offset the other characters)
itos = {i:s for s, i in stoi.items()} #Reverse of stoi

g = torch.Generator().manual_seed(2147483647) #Use a generator so we can get the same random numbers every time

def plot():
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    font_size = 8
    plt.imshow(N, cmap='Blues')

    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray', fontsize=font_size)
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray', fontsize=font_size)

    plt.axis('off')
    plt.show()

#Name Sampling#
#We are 'training' this model using bigrams
def bigram_name_sampling():

    for w in words:
        chs = ['.'] + list(w) + ['.'] #Pairs the starting and ending characters with a token
    for ch1, ch2 in zip(chs, chs[1:]): #Iterates through the bigrams
        ix1 = stoi[ch1] #Storing the bigrams
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 #Counting the occurence of each bigram and reflecting that in the array

    for i in range(50): #Looping multiple times to get sample multiple names
        out = []
        ix = 0 #Set our index to zero so we can get a starting letter (start token .)

        while True:
            #Training
            p = N[ix].float() #Store the row of the index we are currently on
            p = p / p.sum() #Normalize the data so it is in between 0 and 1
            
            #We use multinomial to generate a tensor of integers based on probability : Storing in ix will tell us which index will be next
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() #Example : if the 0th element in the tensor is 0.60, then the tensor should be made up of ~60% 0s

            out.append(itos[ix])
            if ix == 0: #This means we have hit the end token (.)
                break
    def data_view():
        print("==============================================")
        print("VIEWING THE DATA")
        print("==============================================")
        for w in words[:1]:
            print("w as a list = ", list(w))
            chs = ['<S>'] + list(w) + ['<E>']
            for ch1, ch2 in zip(chs, chs[1:]):
                print(ch1, ch2) #Prints the bigram

        chars = sorted(list(set(''.join(words))))
        print('chars = ', chars)

        stoi = {s:i for i, s in enumerate(chars)}
        print('stoi = ', stoi)

        itos = {i:s for s, i in stoi.items()}
        print('itos = ', itos)

        print("First Row = ", N[0])

        print('p = ', p)

        print("Multinomial Funtion = ", torch.multinomial(p, num_samples=100, replacement=True, generator=g))

def bigram_model():#Use the name 'emma' as an example 

    #Create a training set of bigrams
    xs, ys = [], [] #Input data and labels for the input data (what character comes next)

    #Take the first word as an example
    for w in words:
        chs = ['.'] + list(w) + ['.'] #Construct the beginning and end tokens
        for ch1, ch2 in zip(chs, chs[1:]): #Organize the bigrams to get their index
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            #Use the indexes as input and label data
            xs.append(ix1)
            ys.append(ix2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num_examples = xs.nelement()

    print("Number of examples: ", num_examples)

    print('xs: ', xs)
    print('ys: ', ys)

    #Randomly initialze the 27 neurons weights
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True) #Each neuron has 27 inputs
    for k in range(500):
        #Forward pass : All of these layers make it easy for us to backpropogate through
        xenc =F.one_hot(xs, num_classes=27).float() #input to the nn using one hot encoding
        logits = xenc @ W #predict log-counts : multiply and add the x encodings by the weights
        #Softmax : Take ouputs of a nn and outputs probability distributions
        counts = logits.exp() #equivalent to N
        probs = counts / counts.sum(1, keepdims=True) #normalizing distribution : probabilities for next character

        #=====Optimize NN======#
        #Retrieve the loss for the NN : Negative log likelihood
        loss = -probs[torch.arange(num_examples), ys].log().mean()
        print("Loss: ", loss.item())

        #Backward pass
        W.grad = None #Way to set gradients to zero
        loss.backward()

        #Update the weights
        W.data += -0.1 * W.grad #No loop becasue all we have is one tensor

    nlls = torch.zeros(5)
    def run_through():
        for i in range(5):
            #i-th bigram
            x = xs[i].item() #input character index
            y = ys[i].item() #label for x
            print("============")
            print(f'Bigram example {i+1} | {itos[x]}{itos[y]} | (indexes: {x}, {y})')
            print(f'Input to the NN: {itos[x]} | index: {x}')
            print(f'Output probability: {probs[i]}')
            print(f'Label (next character): {itos[y]} | index: {y}')
            p = probs[i, y]
            print(f'Probability assigned by the NN to the correct character: {p.item()}')
            logp = torch.log(p)
            print(f'log likelihood: {logp.item()}')
            nll = -logp
            print(f'negative log likelihood: {nll}')
            nlls[i] = nll
            print("============")

            print(f'Loss/ average negative log likelihood {nlls.mean().item()}')


def MLP_dataset():
    #building the dataset
    def build_dataset(words):
        block_size = 3 #How many characters will we look at before to predict the next one : context length
        X, Y = [], [] #inputs and labels

        for w in words:
            #print(w)
            context = [0] * block_size #current running list of letters

            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                #print(''.join(itos[i] for i in context), '------>', itos[ix])
                context = context[1:] + [ix]
        #converting the lists into tensors for future calculations
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        #print(X.shape, Y.shape)
        return X, Y

    #Splitting the dataset : training - 80%, validation - 10%, testing - 10%
    random.seed(42)
    random.shuffle(words) #shuffling the words in random order

    n1 = int(0.8*len(words)) #80% of the words
    n2 = int(0.9*len(words)) #90% of the words

    Xtrain, Ytrain = build_dataset(words[:n1]) #training
    Xvalidate, Yvalidate = build_dataset(words[n1:n2]) #validation/dev
    Xtest, Ytest = build_dataset(words[n2:]) #testing

    #NOTE: 
    # F.one_hot(5, num_classes=27) #Encoding the integer 5 : Outputs a tensor that is all zeros, except the 5th index which is 1
    # Multiplying the one hot encoding by C will give you the same result as C[5]

    #Creating an embedding look up table
    C = torch.randn((27, 15), generator=g)
    W1 = torch.randn((45, 200), generator=g) #Initializing the weights for our first layer : 6 inputs (3x2) and 100 neurons
    b1 = torch.randn(200, generator=g) #Initializing the biases of the first layer
    #Create the second layer
    W2 = torch.randn((200, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]

    print("Number of parameters: ", sum(p.nelement() for p in parameters)) #total number of parameters

    for p in parameters:
        p.requires_grad = True

    #creating learning rates between 0.001 and 1 (exponentially stepped)
    lre = torch.linspace(-3, 0, 1000) 
    lrs = 10**lre #stepping through the exponents : 10^-3 = 0.001, 10^0 = 1

    #keep track of the learning rates we used and the losses that resulted from them
    lri = []
    lossi = []

    for i in range(500000):

        #constructing the mini batch
        ix = torch.randint(0, Xtrain.shape[0], (64,)) #generate a tensor with numbers between 0 and the shape of x so we can use the integers to index into the dataset

        #Forward Pass
        emb = C[Xtrain[ix]] #Embedding of our input 

        #Change the tensors dimensions so we can multiply
        #Use .view to change the internals of the tensor and keep the same storage properties : -1 is used for pytorch to interpret the appropiate size
        h = torch.tanh(emb.view(-1, 45) @ W1 + b1) #h is the hidden layer of activations

        logits = h @ W2 + b2 #logits are the output of this NN
        #softmax
        # counts = logits.exp()
        # prob = counts / counts.sum(1, keepdims=True)

        #Getting the loss : Negative log likelihood 
        #loss = -prob[torch.arange(32), Y].log().mean() #Comparing with the labels to see the probability of predicting the next character 

        #Witht the cross entropy function, the forward and backward pass are much more efficient, and everything is more numerically well behaved

        loss = F.cross_entropy(logits, Ytrain[ix]) #This calculates the loss using pytorch

        #print(loss.item())

        #Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # lr = lrs[i]

        #update : learning rate decay
        if i < 50000:
            lr = 0.1
        elif i > 50000 and i < 100000:
            lr = 0.05
        else:
            lr = 0.01

        for p in parameters:
            p.data += -lr * p.grad #learning rate x gradient

        # #track stats
        # lri.append(lre[i])
        # lossi.append(loss.item())

    # plt.plot(lri, lossi)
    # plt.show()

    #evaluate the network
    emb = C[Xtrain] 
    h = torch.tanh(emb.view(-1, 45) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytrain)
    print("Training loss: ", loss.item())

    emb = C[Xvalidate] 
    h = torch.tanh(emb.view(-1, 45) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yvalidate)
    print("Validation loss: ", loss.item())

    def sample():
        block_size = 3
        for _ in range(20):
            out = []
            context = [0] * block_size

            while True:
                emb = C[torch.tensor([context])]
                h = torch.tanh(emb.view(1, -1) @ W1 + b1)
                logits = h @ W2 + b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break
                print(''.join(itos[i] for i in out))

    sample()
    
MLP_dataset()




    

