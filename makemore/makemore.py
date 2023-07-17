#Bigram language model - statistical lanugage model that predicts the probabilty of a character/word based on a previous character/word
#Bigram - A piar of consecutive characters/words

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('makemore/names.txt', 'r').read().splitlines() #Puts each name in the text file into a list of words

N = torch.zeros((27, 27), dtype=torch.int32) #Using PyTorch to set a 27x27 (26 letter of the alphabet + start and end token) array with the type of a 32 bit integer

chars = sorted(list(set(''.join(words)))) #storing each character from the alphabet in a list
stoi = {s:i+1 for i, s in enumerate(chars)} #mapping each character to an integer index : offset by 1 to have the '.' as index 0
stoi['.'] = 0 #Instead of having two tokens, we will just use one (Make this position 0 and offset the other characters)
itos = {i:s for s, i in stoi.items()} #Reverse of stoi

for w in words:
    chs = ['.'] + list(w) + ['.'] #Pairs the starting and ending characters with a token
    for ch1, ch2 in zip(chs, chs[1:]): #Iterates through the bigrams
        ix1 = stoi[ch1] #Storing the bigrams
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 #Counting the occurence of each bigram and reflecting that in the array


#+++++++++++++++++++ Name Sampling +++++++++++++++++++#
#We are 'training' this model using bigrams

g = torch.Generator().manual_seed(2147483647) #Use a generator so we can get the same random numbers every time

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
    
    print(''.join(out))

def summary():#Use the name 'emma' as an example

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
    W = torch.randn((27, 27), generator=g, requires_grad=True) #Each neuron has 27 inputs
    for k in range(10):
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


summary()    


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




    

