import torch
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
for word in words:
    chrs = '.' + word + '.'
    for ch1, ch2 in zip(chrs, chrs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print("Number of examples: ",num)

#Initialize network:
gen = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), generator=gen, requires_grad=True)
W.to(device)

#Gradient Descent:
for k in range (100):
    #Forward Pass:
    xEnc = F.one_hot(xs, num_classes=27).float()
    logits = xEnc @ W
    counts = logits.exp() # same as N
    probs = counts / counts.sum(1, keepdim = True) #this is softmax
    loss  = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() #end is regularization term
    print(loss.item())

    #Backward Pass:
    W.grad = None #set gradient to zero
    loss.backward()
    #print(W.grad)

    #Update:
    W.data += -50 * W.grad


for i in range(5):
    ix = 0
    out = []
    while True:
        xEnc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xEnc @ W
        counts = logits.exp() # same as N
        probs = counts / counts.sum(1, keepdim = True) #this is softmax
        ix = torch.multinomial(probs, 1, replacement=True, generator=gen).item()
        out.append(itos[ix])
        if(ix == 0):
            break
    print(''.join(out))
