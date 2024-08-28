import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_data(train, test, dev):
    data = []
    for word_set in [train, test, dev]:
        xs, ys = [], []
        for w in word_set:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                xs.append([stoi[ch1], stoi[ch2]])
                ys.append(stoi[ch3])
        data.append((torch.tensor(xs), torch.tensor(ys)))
    return tuple(data)


def evaluate_onehot(x, y, W):
    xenc = F.one_hot(x, num_classes=27).float().to(device)
    xenc = xenc.view(-1, 27*2)

    logits = xenc @ W
    counts = torch.exp(logits)

    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(len(x)), y].log().mean()

    return loss


def evaluate(x, y, W):
    logits = W[x[:, 0]] + W[x[:, 1] + 27]
    loss = F.cross_entropy(logits, y.to(device))
    return loss


def train(x_train, y_train, x_dev, y_dev):
    g = torch.Generator(device).manual_seed(2147483647)
    W = torch.randn((27*2, 27), generator=g, requires_grad=True, device=device)

    for k in range(100):
        loss = evaluate(x_train, y_train, W) + 0.2 * (W**2).mean()
        W.grad = None
        loss.backward()

        with torch.no_grad():
            W.data += -50 * W.grad

    print(
        f"loss: {loss.item():.4f} | Dev loss: {evaluate(x_dev, y_dev, W).item():.4f}")
    return W


def sample_onehot(W):
    # Generate 10 new names
    names = []
    for i in range(10):
        out = []
        ix1, ix2 = 0, 0  # Start with two padding characters
        while True:
            # Create one-hot encoded input
            xenc = F.one_hot(torch.tensor([ix1, ix2]).to(
                device), num_classes=27).float().to(device)
            xenc = xenc.view(-1, 27*2)

            # Get logits from the model
            logits = xenc @ W

            # Convert logits to probabilities
            counts = torch.exp(logits)
            p = counts / counts.sum(dim=1, keepdims=True)

            # Shift characters and sample next character
            ix1 = ix2
            ix2 = torch.multinomial(
                p.to(device), num_samples=1, replacement=True).item()

            if ix2 == 0:  # End of name reached
                break
            out.append(itos[ix2])

        names.append("".join(out))

    return names


def sample(W):
    names = []
    for _ in range(10):
        out = []
        ix = [0, 0]
        while True:
            logits = W[ix[0]] + W[ix[1] + 27]
            p = F.softmax(logits, dim=0)
            ix = [ix[1], torch.multinomial(p, num_samples=1).item()]
            if ix[1] == 0:
                break
            out.append(itos[ix[1]])
        names.append("".join(out))
    return names


if __name__ == "__main__":
    with open('names.txt', 'r') as f:
        words = f.read().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}

    words_train, words_test = train_test_split(words, test_size=0.2,
                                               random_state=1234)
    words_dev, words_test = train_test_split(words_test, test_size=0.5,
                                             random_state=1234)

    (x_train, y_train), (x_test, y_test), (x_dev, y_dev) = build_data(
        train=words_train, test=words_test, dev=words_dev)

    W = train(x_train, y_train, x_dev, y_dev)

    loss = evaluate(x_test, y_test, W)
    print(f"X test log:  {loss.item()}")

    loss = evaluate(x_dev, y_dev, W)
    print(f"X dev log:  {loss.item()}")

    names = sample(W)
    # Print generated names
    for name in names:
        print(name)
