#Bigram language model - statistical lanugage model that predicts the probabilty of a character/word based on a previous character/word
#Bigram - A piar of consecutive characters/words

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('Practical-Deep-Learning/makemore/names.txt', 'r').read().splitlines() #Puts each name in the text file into a list of words

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


#Create a matrix that has the probabilities of the bigrams already caluclated so we do not have to do it in the for loop below
P = (N+1).float() #Getting the float values of N : Adding to N to smooth our model. Some bigrams are 0% likely, which if seen in a testing set will make our loss infinity
P /= P.sum(1, keepdim=True) #If we pass in the dimension and set keepdim = True, we can sum up the separate rows in our overall matrix : See broadcasting semantics to see why this operation is possible

#When we normalize across the rows, we expect P[0].sum() to always be 1

def single_layer():
    # Create the training set of bigrams
    xs, ys = [], []
    for w in words[:1]: #Only looking at the first words : We have 5 inputs into the nn
        chs = ['.'] + list(w) + ['.'] 
        for ch1, ch2 in zip(chs, chs[1:]): 
            ix1 = stoi[ch1] 
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

         
    #xs are the input to the nn, ys are the label
    xs = torch.tensor(xs) #Creating tensors for the input and output of the nn : xs and ys are the index of the characters. Example: a=1, b=2, .=0
    ys = torch.tensor(ys) 

    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g) #Randomnly initialize 27 neurons weights : Each neuron receives 27 inputs 

    #Forward pass
    xenc = F.one_hot(xs, num_classes=27).float() #input to nn using one-hot encoding : need to cast to float to have easier tweaked values
    logits = (xenc @ W) #predict log-counts : @ is a matrix multiplication operator : (5, 27) @ (27, 27) ===> output (5, 27) : This represents the 5 activations on 27 neuron based on the weighted inputs
    counts = logits.exp() #Equivalent to N : We exponentiate this output so we can utilize it when looking at probability : This is the output for the one layer nn
    probs = counts / counts.sum(1, keepdims=True) #Normalize the probabilities : Every rows will now sum to 1 : shape is (5, 27)

    #The 2 lines above are called a softmax : Softmax is a way to take an output from a nn and covert it into probabilities
    
    #.e is the first input in our nn : Having the probs gives us the probability of each character coming next, sinces there are 27 indexes in each row
    def view_data():
        print('xs: ', xs)
        print('ys: ', ys) 

        print('xenc: ', xenc) 

        print('logits: ', logits)
        print('counts: ', counts)
        print('probs: ', probs)

    def example():
        nlls = torch.zeros(5)

        for i in range(5):
            #i-th bigram
            x = xs[i].item()
            y = ys[i].item()
            print('------------------')
            print(f'Bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x}, {y})')
            print('INput to the neural net: ', probs[i])
            print('label (actual next character): ', y)
            p = probs[i, y]
            print('Probability assigned by the net to the correct character: ', p.item())
            logp = torch.log(p)
            print('Log likelihood: ', logp.item())
            nll = -logp
            print('Negative log likelihood: ', nll.item())
            nlls[i] = nll

        print('=================')
        print('Average Negative log likelihood, i.e. loss = ', nlls.mean().item())

    example()

single_layer()

#+++++++++++++++++++ Name Sampling +++++++++++++++++++#
#We are 'training' this model using bigrams
def name_sample():
    g = torch.Generator().manual_seed(2147483647) #Use a generator so we can get the same random numbers every time

    for i in range(5): #Looping multiple times to get sample multiple names
        out = []
        ix = 0 #Set our index to zero so we can get a starting letter (start token .)

        while True:
            #Training
            p = P[ix]
            # p = N[ix].float() #Store the row of the index we are currently on
            # p = p / p.sum() #Normalize the data so it is in between 0 and 1
            
            #We use multinomial to generate a tensor of integers based on probability : Storing in ix will tell us which index will be next
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() #Example : if the 0th element in the tensor is 0.60, then the tensor should be made up of ~60% 0s

            out.append(itos[ix])
            if ix == 0: #This means we have hit the end token (.)
                break
        
        print(''.join(out))

def view_model_quality():
    #GOAL: Maximize likelihood (product of the probablities) of the data
    #Equivalent to maximizing the log likelihood (This is just scaling)
    #Equivalent to minimizing the negative log likelihood
    #Equivalent to minimizing the average negative log likelihood




    #To measure the quality of our model as one number, we can take the product of all of the probabilities in the training set
    #Instead, we will use the log likelihood for convenience
    #log(a*b*c) = log(a) + log(b) + log(c)

    log_likelihood = 0.0
    n = 0 #variable for normalization
    for w in ['harrison']: #Looking at the probability of the bigrams in my name : be careful, if you test aname with a bigram that has probability = 0, you will get a bug
        chs = ['.'] + list(w) + ['.'] 
        for ch1, ch2 in zip(chs, chs[1:]): 
            ix1 = stoi[ch1] 
            ix2 = stoi[ch2]
            prob = P[ix1, ix2] #Looks up the probability of the bigram according to the matrix P 
            logprob = torch.log(prob) #Gets the log probability (all of the log products summed together)
            log_likelihood += logprob #Accumulates the log probabilities
            n += 1
            print(f'{ch1}{ch2}: {prob:.4f}: {logprob:.4f}')

    print(f'{log_likelihood=}') #When all of the probabilities are 1, log_likelihood will be zero. When the proabilities are lower, this number grows more and more negative
    nll = -log_likelihood
    print(f'{nll=}') #This is a nice loss function because the lowest this can get is 0, and the worse off the probabilities, the higher the number
    print(f'{nll/n}') #Average log likelihood (This can be our loss function). Therefore, the qualioty of our model is this output


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




    

