#Bigram language model - statistical lanugage model that predicts the probabilty of a character/word based on a previous character/word
#Bigram - A piar of consecutive characters/words

import torch
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




    

