import torch
import random
import torch.nn.functional as F

block_size = 3  # How many chars taken to predict the next one
n_embd = 10  # The dimensionality of the character embedding vectors
n_hidden = 200  # The number of neurons in the hidden layer of the MLP
max_steps = 200000
batch_size = 32
lossi = []


def sample():
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            # Forward pass neural net
            emb = C[torch.tensor([context]).cuda()]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1).cuda()
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1).cuda()

            # Sample from distribution
            ix = torch.multinomial(probs, num_samples=1).cuda().item()

            # Shift context window
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))


@torch.no_grad()
def split_loss(split, bnmean, bnstd):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[x]  # (M. block_size, n_embd)
    # concat into (N, block_size * n_embed)
    embcat = emb.view(emb.shape[0], -1).cuda()
    hpreact = embcat @ W1 + b1

    # Batch Normalisation
    hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias

    h = torch.tanh(hpreact)  # (N, n_hidden)
    logits = h @ W2 + b2  # (N, vocab_size)
    loss = F.cross_entropy(logits, y).cuda()
    print(split, loss.cuda().item())


def build_dataset(words):
    X, Y = [], []  # X is the input, Y is the labels

    for w in words:
        context = [0] * block_size

        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X).cuda()
    Y = torch.tensor(Y).cuda()

    return X, Y


if __name__ == "__main__":
    with open('names.txt', 'r') as f:
        words = f.read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(itos)

    random.seed(42)
    random.shuffle(words)

    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    g = torch.Generator().manual_seed(2147463647)
    C = torch.randn((vocab_size, n_embd), generator=g).cuda()
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g).cuda() * 0.2
    W2 = torch.randn((n_hidden, vocab_size), generator=g).cuda() * \
        0.01  # Don't want to set this to exactly 0
    b2 = torch.randn(vocab_size, generator=g).cuda() * 0

    bngain = torch.ones((1, n_hidden)).cuda()
    bnbias = torch.zeros((1, n_hidden)).cuda()

    bnmean_running = torch.zeros((1, n_hidden)).cuda()
    bnstd_running = torch.ones((1, n_hidden)).cuda()

    parameters = [C, W1, W2, b2, bngain, bnbias]
    for p in parameters:
        p.requires_grad = True

    for i in range(max_steps):

        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g).cuda()
        Xb, Yb = Xtr[ix], Ytr[ix]  # Batch X, Y

        # forward pass
        emb = C[Xb]  # Embed characters into vectors
        embcat = emb.view(emb.shape[0], -1).cuda()  # Concatenate the vectors
        hpreact = embcat @ W1

        bnmeani = hpreact.mean(0, keepdim=True).cuda()
        bnstdi = hpreact.std(0, keepdim=True).cuda()

        # Batch Normalisation
        hpreact = bngain * (hpreact - bnmeani) / \
            bnstdi + bnbias

        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

        h = torch.tanh(hpreact).cuda()  # Hidden layer
        logits = h @ W2 + b2  # Output layer
        loss = F.cross_entropy(logits, Yb).cuda()  # Loss function

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        if i % 10000 == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())

    split_loss('train', bnmean_running, bnstd_running)
    split_loss('val', bnmean_running, bnstd_running)
