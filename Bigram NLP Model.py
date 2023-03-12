#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


names = open("names.txt", "r").read().splitlines()


# In[3]:


names[:5]


# In[4]:


bigrams = {}
for w in names:
    chars = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chars, chars[1:]):
        bigram = (ch1, ch2)
        bigrams[bigram] = bigrams.get(bigram, 0) + 1


# In[5]:


bigrams = sorted(bigrams.items(), key=lambda x: -x[1]) # sort according to bigram frequency


# In[6]:


N = torch.ones((27, 27), dtype=torch.int32)


# In[7]:


N


# In[8]:


# Instead of a dictionary, we are going to use a 2x2 matrix to store the results of the dataset.
# In order to do this, we need some way to convert from characters to indices. This will be done using a lookup table,
# defined below.


# In[9]:


alphabet = sorted(list(set(''.join(names))))

charToInt = {c: i + 1 for i, c in enumerate(alphabet)}
charToInt['.'] = 0

intToChar = {i: c for c, i in charToInt.items()}


# In[10]:


for name in names:
    chars = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        indexOne = charToInt[ch1]
        indexTwo = charToInt[ch2]
        
        N[indexOne, indexTwo] += 1


# In[11]:


plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        characterPair = intToChar[i] + intToChar[j]
        plt.text(j, i, characterPair + '\n', ha='center', va = 'center')
        plt.text(j, i, int(N[i][j]), ha='center', va = 'top')

plt.axis('off');


# In[12]:


probabilityDist = N.float() / N.sum(1, keepdim=True)


# In[13]:


g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    out = []
    index = 0
    while True:
        probability = probabilityDist[index].float()
        index = torch.multinomial(probability, num_samples=1, replacement=True, generator=g).item()
        out.append(intToChar[index])
        if index == 0:
            break
        
    print(''.join(out))


# In[14]:


# Introducing negative log likelihood 
pairCount = 0
logLikelihood = 0.0

for w in names:
    chars = ['.'] + list(w) + ['.']
    
    for charOne, charTwo in zip(chars, chars[1:]):
        pairCount += 1
        
        indexOne = charToInt[charOne]
        indexTwo = charToInt[charTwo]
        
        prob = probabilityDist[indexOne, indexTwo]
        
        logProb = torch.log(prob)
        logLikelihood += logProb
        

negLoglikelihood = -logLikelihood
print(negLoglikelihood / pairCount)


# In[15]:


print(names[0:10])


# In[21]:


xs, ys = [], []

for w in names[:1]:
    chars = ['.'] + list(w) + ['.']
    
    for charOne, charTwo in zip(chars, chars[1:]):        
        indexOne = charToInt[charOne]
        indexTwo = charToInt[charTwo]
        
        xs.append(indexOne)
        ys.append(indexTwo)
        
        print(charOne, charTwo)
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(xs, ys)


# In[29]:


import torch.nn.functional as F
xEnc = F.one_hot(xs, num_classes=27).float()


# In[31]:


plt.imshow(xEnc, cmap='Blues')


# In[117]:


g = torch.Generator().manual_seed(2147483647)

layer = torch.randn((27, 27), generator=g, requires_grad=True)


# In[140]:


# create dataset
xs, ys = [], []

for name in names:
    characters = ['.'] + list(name) + ['.']
    
    for charOne, charTwo in zip(characters, characters[1:]):
        xs.append(charToInt[charOne])
        ys.append(charToInt[charTwo])

inputs = torch.tensor(xs)
outputs = torch.tensor(ys)

datasetLength = len(inputs)


# In[141]:


inputsEncoded = F.one_hot(inputs, num_classes=27).float()
outputsEncoded = F.one_hot(outputs, num_classes=27).float()


# In[154]:


for i in range(100):
    
    # forward pass
    logits = inputsEncoded @ layer

    probs = logits.exp()
    normalised = probs / probs.sum(1, keepdims=True)
    
    negLogLikelihood = -normalised[torch.arange(datasetLength), outputs].log().mean()
    print(negLogLikelihood.item())
    
    # backward pass
    layer.grad = None
    negLogLikelihood.backward()
    
    # update
    layer.data += -50 * layer.grad
    
