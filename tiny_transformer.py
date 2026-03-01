import numpy as np

# ==========================
# Hyperparameters
# ==========================
embedding_dim = 32
ff_dim = 64
learning_rate = 0.001
epochs = 120
max_len = 10
temperature = 0.8
clip_value = 5.0

# ==========================
# Load Dataset
# ==========================
sentences = []
with open("dataset.txt", "r") as f:
    for line in f:
        line = line.strip().lower()
        if line:
            sentences.append(line)

# ==========================
# Build Vocabulary
# ==========================
vocab = set()
for s in sentences:
    vocab.update(s.split())

vocab = list(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)

# ==========================
# Initialize Parameters
# ==========================
np.random.seed(42)

E = np.random.randn(vocab_size, embedding_dim) * 0.1

Wq = np.random.randn(embedding_dim, embedding_dim) * 0.1
Wk = np.random.randn(embedding_dim, embedding_dim) * 0.1
Wv = np.random.randn(embedding_dim, embedding_dim) * 0.1

W1 = np.random.randn(embedding_dim, ff_dim) * 0.1
W2 = np.random.randn(ff_dim, embedding_dim) * 0.1

Wo = np.random.randn(embedding_dim, vocab_size) * 0.1

# ==========================
# Helpers
# ==========================
def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

def clip(g):
    return np.clip(g, -clip_value, clip_value)

# ==========================
# Training
# ==========================
for epoch in range(epochs):

    total_loss = 0.0

    for sentence in sentences:
        words = sentence.split()

        for i in range(1, len(words)):

            context = words[:i]
            target_word = words[i]

            X = np.array([E[word_to_idx[w]] for w in context])

            # ===== FORWARD =====

            Q = X @ Wq
            K = X @ Wk
            V = X @ Wv

            scores = Q @ K.T / np.sqrt(embedding_dim)
            exp_scores = np.exp(scores - np.max(scores))
            A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            Z = A @ V

            FF1 = np.tanh(Z @ W1)
            FF2 = FF1 @ W2

            logits = FF2[-1] @ Wo
            probs = softmax(logits)

            target_idx = word_to_idx[target_word]
            loss = -np.log(probs[target_idx] + 1e-9)
            total_loss += loss

            # ===== BACKPROP =====

            # Output gradient
            d_logits = probs.copy()
            d_logits[target_idx] -= 1

            dWo = np.outer(FF2[-1], d_logits)
            Wo -= learning_rate * clip(dWo)

            # Backprop into FF2
            dFF2_last = Wo @ d_logits

            # W2
            dW2 = np.outer(FF1[-1], dFF2_last)
            W2 -= learning_rate * clip(dW2)

            # Backprop into FF1
            dFF1_last = dFF2_last @ W2.T

            # tanh derivative
            dPreFF1 = dFF1_last * (1 - FF1[-1]**2)

            # Backprop into Z
            dZ_last = dPreFF1 @ W1.T

            # W1
            dW1 = np.outer(Z[-1], dPreFF1)
            W1 -= learning_rate * clip(dW1)

            # ===== ATTENTION BACKPROP =====

            dZ = np.zeros_like(Z)
            dZ[-1] = dZ_last

            # Z = A @ V
            dA = dZ @ V.T
            dV = A.T @ dZ

            # Softmax backward (row-wise)
            dScores = np.zeros_like(scores)

            for r in range(A.shape[0]):
                a = A[r]
                jacobian = np.diag(a) - np.outer(a, a)
                dScores[r] = jacobian @ dA[r]

            dScores /= np.sqrt(embedding_dim)

            # scores = QKᵀ
            dQ = dScores @ K
            dK = dScores.T @ Q

            # Q = XWq etc.
            dWq = X.T @ dQ
            dWk = X.T @ dK
            dWv = X.T @ dV

            Wq -= learning_rate * clip(dWq)
            Wk -= learning_rate * clip(dWk)
            Wv -= learning_rate * clip(dWv)

            # ===== EMBEDDING BACKPROP =====

            dX_q = dQ @ Wq.T
            dX_k = dK @ Wk.T
            dX_v = dV @ Wv.T

            dX = dX_q + dX_k + dX_v

            for idx_pos, word in enumerate(context):
                word_idx = word_to_idx[word]
                E[word_idx] -= learning_rate * clip(dX[idx_pos])

    if epoch % 20 == 0:
        print("Epoch", epoch, "Loss:", float(total_loss))

print("\nTraining Complete!\n")

# ==========================
# Generation
# ==========================
def generate(start_word, length=15):

    words = [start_word]

    for _ in range(length):

        context = words[-max_len:]
        X = np.array([E[word_to_idx[w]] for w in context if w in word_to_idx])

        if len(X) == 0:
            break

        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv

        scores = Q @ K.T / np.sqrt(embedding_dim)
        exp_scores = np.exp(scores - np.max(scores))
        A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        Z = A @ V

        FF1 = np.tanh(Z @ W1)
        FF2 = FF1 @ W2

        logits = FF2[-1] @ Wo
        logits = logits / temperature

        probs = softmax(logits)

        next_idx = np.random.choice(len(probs), p=probs)
        next_word = idx_to_word[next_idx]

        words.append(next_word)

    return " ".join(words)

# ==========================
# Chat Loop
# ==========================
while True:
    user = input("Start word: ").lower()
    if user not in word_to_idx:
        print("Word not in vocabulary.")
        continue
    print("Generated:", generate(user))