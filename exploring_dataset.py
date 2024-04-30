import torch
import matplotlib
print(matplotlib.get_backend())
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

with open('names.txt', 'r') as f:
    words = f.read().splitlines()

#print(words[:100])

#now we print character bigrams


N = torch.zeros((28,28), dtype = torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = { ch:i for i, ch in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

bigrams = {}
for word in words:
    characters = ['<S>'] + list(word) + ['<E>']
    for ch1, ch2 in zip(characters, characters[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1
        #print(ch1,ch2)


plt.imshow(N)