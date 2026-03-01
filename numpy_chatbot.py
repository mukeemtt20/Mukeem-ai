import numpy as np

# ==========================
# Hyperparameters
# ==========================
embedding_dim = 16
hidden_dim = 64
learning_rate = 0.01
epochs = 300

END = "<END>"

# ==========================
# Load Dataset
# ==========================
pairs = []

with open("dataset.txt", "r") as f:
    for line in f:
        if "|" in line:
            q, a = line.strip().split("|")
            pairs.append((q.lower(), a.lower()))

# ==========================
# Build Vocabulary
# ==========================
vocab = set()

for q, a in pairs:
    vocab.update(q.split())
    vocab.update(a.split())

vocab.add(END)

vocab = list(vocab)

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

vocab_size = len(vocab)

# ==========================
# Initialize Parameters
# ==========================
np.random.seed(42)

E = np.random.randn(vocab_size, embedding_dim) * 0.1

Wxh = np.random.randn(hidden_dim, embedding_dim) * 0.1
Whh = np.random.randn(hidden_dim, hidden_dim) * 0.1
Why = np.random.randn(vocab_size, hidden_dim) * 0.1

bh = np.zeros((hidden_dim, 1))
by = np.zeros((vocab_size, 1))

# ==========================
# Helper Functions
# ==========================
def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

# ==========================
# Training
# ==========================
for epoch in range(epochs):

    total_loss = 0.0

    for question, answer in pairs:

        h_prev = np.zeros((hidden_dim, 1))

        # -------- Encoder --------
        for word in question.split():
            if word not in word_to_idx:
                continue

            x = E[word_to_idx[word]].reshape(-1, 1)
            h_prev = np.tanh(Wxh @ x + Whh @ h_prev + bh)

        # -------- Decoder --------
        target_words = answer.split() + [END]

        for word in target_words:

            logits = Why @ h_prev + by
            probs = softmax(logits)

            target_idx = word_to_idx[word]
            loss = -np.log(float(probs[target_idx]) + 1e-9)
            total_loss += loss

            # Gradient
            d_logits = probs.copy()
            d_logits[target_idx] -= 1

            # Update output layer
            Why -= learning_rate * (d_logits @ h_prev.T)
            by -= learning_rate * d_logits

            # Backprop to hidden
            dh = Why.T @ d_logits
            dh_raw = (1 - h_prev ** 2) * dh

            Whh -= learning_rate * (dh_raw @ h_prev.T)
            bh -= learning_rate * dh_raw

            # Update hidden
            h_prev = np.tanh(Whh @ h_prev + bh)

    if epoch % 50 == 0:
        print("Epoch", epoch, "Loss:", float(total_loss))

print("\nTraining Complete!\n")

# ==========================
# Chat Function
# ==========================
def reply(sentence):

    h_prev = np.zeros((hidden_dim, 1))

    # Encode
    for word in sentence.lower().split():
        if word not in word_to_idx:
            continue

        x = E[word_to_idx[word]].reshape(-1, 1)
        h_prev = np.tanh(Wxh @ x + Whh @ h_prev + bh)

    # Decode
    response = []

    for _ in range(15):

        logits = Why @ h_prev + by
        probs = softmax(logits)

        next_idx = int(np.argmax(probs))
        next_word = idx_to_word[next_idx]

        if next_word == END:
            break

        response.append(next_word)

        x = E[next_idx].reshape(-1, 1)
        h_prev = np.tanh(Wxh @ x + Whh @ h_prev + bh)

    return " ".join(response)


# ==========================
# Chat Loop
# ==========================
while True:
    user = input("You: ")
    print("Bot:", reply(user))