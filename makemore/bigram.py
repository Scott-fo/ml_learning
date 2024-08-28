import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for i, s in enumerate(stoi)}

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()


# Randomly initialize 27 neurons weights. each neuron has 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# Gradient Descent
for k in range(100):
    # Forward Pass
    # Input to netork.
    # This makes a vector where the value is a 1 in the nth column
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W  # Predict log counts

    # Softmax
    counts = logits.exp()  # Counts, equivalent to N
    probs = counts / counts.sum(1, keepdims=True)  # Probs for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    print(loss.item())

    W.grad = None
    loss.backward()

    W.data += -100 * W.grad


def notes():
    nlls = torch.zeros(5)
    for i in range(5):
        x = xs[i].item()
        y = ys[i].item()
        print("-------")
        print(f'bigram example {i + 1}: {itos[x]}{itos[y]} (indexes {x},{y})')
        print('input to the neural net:', x)
        print('output probabilities from the neural net:', probs[i])
        print('label (actual next character): ', y)
        p = probs[i, y]
        print('probablility assigned by the net to the correct character: ',
              p.item())
        logp = torch.log(p)
        print('log likelihood: ', logp.item())
        nll = -logp
        print('negative log likelihood: ', nll.item())
        nlls[i] = nll

    print('======')
    print('average negative log likelihood, ie loss = ', nlls.mean().item())
