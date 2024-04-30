import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

chars = sorted(list(set(''.join(words))))
stoi = { ch:i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0

itos = { i: ch for ch, i in stoi.items()}

xs, ys = [], []
for word in words[:1]:
    chrs = '.' + word + '.'
    for ch1, ch2 in zip(chrs, chrs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xEnc = F.one_hot(xs, num_classes=27).float()

W = torch.randn((27,27))

logits = xEnc @ W
counts = logits.exp() # same as N
probs = counts / counts.sum(1, keepdim = True) #this is softmax


#single logliklihood example
# nlls = torch.zeros(5)
# for i in range(5):
#     x = xs[i].item()
#     y = ys[i].item()
#     print('------------')
#     print(f'Bigram example {i}: {itos[x]} {itos[y]}')
#     print(f'Input to NN: {x}')
#     print(f'Ouput probability: {probs[i]}')
#     print(f'Label actual next char: {y}')
#     p = probs[i,y]
#     print(f'Probability assgined to next correct char: {p}')
#     logliklihood = torch.log(p)
#     print(f'Logliklihood: {logliklihood}')
#     print(f'Negative Logliklihood: {-logliklihood}')
#     nlls[i] = -logliklihood

# print('===================')
# print('average negative logliklihood, (loss): ', nlls.mean().item())



##OPTIMIZATION:
gen = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=gen, requires_grad=True)


#Forward Pass:
xEnc = F.one_hot(xs, num_classes=27).float()
logits = xEnc @ W
counts = logits.exp() # same as N
probs = counts / counts.sum(1, keepdim = True) #this is softmax
loss  = -probs[torch.arange(5), ys].log().mean()
print(loss.item())


#Backward Pass:
W.grad = None #set gradient to zero
loss.backward()
#print(W.grad)

#Update:
W.data += -0.1 * W.grad

xEnc = F.one_hot(xs, num_classes=27).float()
logits = xEnc @ W
counts = logits.exp() # same as N
probs = counts / counts.sum(1, keepdim = True) #this is softmax
loss  = -probs[torch.arange(5), ys].log().mean()
print(loss.item())




