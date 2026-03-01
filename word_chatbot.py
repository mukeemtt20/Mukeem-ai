import random
import math

# --------------------------
# Activation
# --------------------------

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - x*x

def softmax(vec):
    max_val = max(vec)
    exps = [math.exp(v - max_val) for v in vec]
    total = sum(exps)
    return [e/total for e in exps]

# --------------------------
# Load Dataset
# --------------------------

pairs = []

with open("dataset.txt", "r") as f:
    for line in f:
        if "|" in line:
            q, a = line.strip().split("|")
            pairs.append((q.lower(), a.lower()))

# Add START and END tokens
START = "<START>"
END = "<END>"

# --------------------------
# Build Vocabulary
# --------------------------

vocab = set()

for q, a in pairs:
    vocab.update(q.split())
    vocab.update(a.split())

vocab.add(START)
vocab.add(END)

vocab = list(vocab)
word_to_idx = {w:i for i,w in enumerate(vocab)}
idx_to_word = {i:w for w,i in word_to_idx.items()}

vocab_size = len(vocab)
hidden_size = 32
learning_rate = 0.05

def one_hot(index):
    vec = [0]*vocab_size
    vec[index] = 1
    return vec

# --------------------------
# Initialize Weights
# --------------------------

Wxh = [[random.uniform(-0.5,0.5) for _ in range(vocab_size)] for _ in range(hidden_size)]
Whh = [[random.uniform(-0.5,0.5) for _ in range(hidden_size)] for _ in range(hidden_size)]
Why = [[random.uniform(-0.5,0.5) for _ in range(hidden_size)] for _ in range(vocab_size)]

bh = [0]*hidden_size
by = [0]*vocab_size

# --------------------------
# Training
# --------------------------

for epoch in range(1500):

    total_loss = 0

    for question, answer in pairs:

        # -------- ENCODER --------
        h_prev = [0]*hidden_size

        for word in question.split():
            if word not in word_to_idx:
                continue

            x = one_hot(word_to_idx[word])

            h = []
            for i in range(hidden_size):
                sum_input = sum(Wxh[i][j]*x[j] for j in range(vocab_size))
                sum_hidden = sum(Whh[i][j]*h_prev[j] for j in range(hidden_size))
                h.append(tanh(sum_input + sum_hidden + bh[i]))

            h_prev = h

        # -------- DECODER --------
        target_words = [START] + answer.split() + [END]

        for word in target_words:

            x = h_prev

            raw_output = []
            for i in range(vocab_size):
                sum_out = sum(Why[i][j]*x[j] for j in range(hidden_size))
                raw_output.append(sum_out + by[i])

            y = softmax(raw_output)

            target_index = word_to_idx[word]
            loss = -math.log(y[target_index] + 1e-9)
            total_loss += loss

            dy = y[:]
            dy[target_index] -= 1

            # Update output weights
            for i in range(vocab_size):
                for j in range(hidden_size):
                    Why[i][j] -= learning_rate * dy[i] * x[j]
                by[i] -= learning_rate * dy[i]

            # Update hidden state
            h_new = []
            for i in range(hidden_size):
                sum_hidden = sum(Whh[i][j]*h_prev[j] for j in range(hidden_size))
                h_new.append(tanh(sum_hidden + bh[i]))

            h_prev = h_new

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("\nTraining Complete!\n")

# --------------------------
# Chat Function
# --------------------------

def reply(sentence):

    h_prev = [0]*hidden_size

    # Encode
    for word in sentence.lower().split():
        if word not in word_to_idx:
            continue

        x = one_hot(word_to_idx[word])

        h = []
        for i in range(hidden_size):
            sum_input = sum(Wxh[i][j]*x[j] for j in range(vocab_size))
            sum_hidden = sum(Whh[i][j]*h_prev[j] for j in range(hidden_size))
            h.append(tanh(sum_input + sum_hidden + bh[i]))

        h_prev = h

    # Decode
    output_sentence = []

    for _ in range(20):

        raw_output = []
        for i in range(vocab_size):
            sum_out = sum(Why[i][j]*h_prev[j] for j in range(hidden_size))
            raw_output.append(sum_out + by[i])

        y = softmax(raw_output)

        next_index = y.index(max(y))
        next_word = idx_to_word[next_index]

        if next_word == END:
            break

        if next_word != START:
            output_sentence.append(next_word)

        # Update hidden
        h_new = []
        for i in range(hidden_size):
            sum_hidden = sum(Whh[i][j]*h_prev[j] for j in range(hidden_size))
            h_new.append(tanh(sum_hidden + bh[i]))

        h_prev = h_new

    return " ".join(output_sentence)


while True:
    user = input("You: ")
    print("Bot:", reply(user))