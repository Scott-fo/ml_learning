import torch
import random
import torch.nn.functional as F


def build_dataset(words):
    block_size = 3  # How many chars taken to predict the next one
    X, Y = [], []  # X is the input, Y is the labels

    for w in words:
        # print(w)
        context = [0] * block_size

        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '---->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X).cuda()
    Y = torch.tensor(Y).cuda()

    return X, Y


def sample(block_size):
    for _ in range(20):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(itos[i] for i in out))


if __name__ == "__main__":
    with open('names.txt', 'r') as f:
        words = f.read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}

    random.seed(42)
    random.shuffle(words)

    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    g = torch.Generator().manual_seed(2147463647)

    C = torch.empty((27, 10)).cuda()
    torch.nn.init.normal_(C, mean=0, std=0.1)

    W1 = torch.empty((30, 200)).cuda()
    torch.nn.init.normal_(W1, mean=0, std=0.1)
    b1 = torch.zeros(200).cuda()

    W2 = torch.empty((200, 27)).cuda()
    torch.nn.init.normal_(W2, mean=0, std=0.1)
    b2 = torch.zeros(27).cuda()

    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
        p.requires_grad = True

    lre = torch.linspace(-3, 0, 1000)
    lrs = 10**lre

    lri = []
    lossi = []
    stepi = []

    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev).cuda()

    print(f"starting loss: {loss.item()}")

    for i in range(200000):
        if i % 100 == 0:
            print(i)

        ix = torch.randint(0, Xtr.shape[0], (32, )).cuda()
        # Forward pass
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1).cuda()
        logits = h @ W2 + b2

        loss = F.cross_entropy(logits, Ytr[ix]).cuda()

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # Stats
        # We can plot lri against lossi to determine what the best learning rate
        # is
        # lri.append(lre[i])
        lossi.append(loss.log10().item())
        stepi.append(i)

    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev).cuda()

    print(loss.item())
