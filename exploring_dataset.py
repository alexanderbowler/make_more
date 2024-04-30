import torch
import matplotlib.pyplot as plt

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

with open('names.txt', 'r') as f:
    words = f.read().splitlines()

#print(words[:100])

#now we print character bigrams


N = torch.zeros((27,27), dtype = torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = { ch:i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0

itos = { i: ch for ch, i in stoi.items()}

bigrams = {}
for word in words:
    characters = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(characters, characters[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1
        #print(ch1,ch2)


plt.figure(figsize = (16,16))
plt.imshow(N, cmap = "Blues")
for i in range(27):
    for j in range(27):
        chstrs = itos[i]+itos[j]
        plt.text(j,i, chstrs, ha = 'center', va = "bottom", color = 'gray')
        plt.text(j,i, N[i,j].item(), ha = 'center', va = 'top', color = 'gray')
plt.axis('off')


gen = torch.Generator(device=device)
gen = gen.manual_seed(2147483647)

P = (N+1).float()/N.sum(1,keepdim=True)
P = P.to(device)

for i in range(5):
    ix = 0
    out = []
    while True:
        prob = P[ix]
        ix = torch.multinomial(prob, 1, replacement=True, generator=gen).item()
        out.append(itos[ix])
        if(ix == 0):
            break
    print(''.join(out))

logliklihood = 0.0
n = 0
for word in words:
    characters = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(characters, characters[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        logliklihood += logprob
        n+=1
        #print(f'{ch1}{ch2}: {prob :.4f} {logprob: .4f}')
        #print(ch1,ch2)

print(logliklihood.item())
negative_log_liklihood = -logliklihood
print(f'{negative_log_liklihood=}')
#normalized
print(f'{negative_log_liklihood/n}')


#plt.show()