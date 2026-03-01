import random
import math

# ------------------------
# Activation
# ------------------------

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - x*x

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ------------------------
# Training Text
# ------------------------

text = "hello hi hey hello hey hi "

chars = list(set(text))
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for ch,i in char_to_idx.items()}

vocab_size = len(chars)
hidden_size = 16
learning_rate = 0.05

def one_hot(index):
    vec = [0] * vocab_size
    vec[index] = 1
    return vec

# ------------------------
# Initialize Weights
# ------------------------

# Input → Hidden
Wxh = [[random.uniform(-0.5,0.5) for _ in range(vocab_size)] for _ in range(hidden_size)]

# Hidden → Hidden
Whh = [[random.uniform(-0.5,0.5) for _ in range(hidden_size)] for _ in range(hidden_size)]

# Hidden → Output
Why = [[random.uniform(-0.5,0.5) for _ in range(hidden_size)] for _ in range(vocab_size)]

bh = [0]*hidden_size
by = [0]*vocab_size

# ------------------------
# Training
# ------------------------

for epoch in range(1500):

    total_loss = 0
    h_prev = [0]*hidden_size

    for t in range(len(text)-1):

        x = one_hot(char_to_idx[text[t]])
        target = one_hot(char_to_idx[text[t+1]])

        # ---- Forward ----

        h = []
        for i in range(hidden_size):
            sum_input = sum(Wxh[i][j]*x[j] for j in range(vocab_size))
            sum_hidden = sum(Whh[i][j]*h_prev[j] for j in range(hidden_size))
            h.append(tanh(sum_input + sum_hidden + bh[i]))

        y = []
        for i in range(vocab_size):
            sum_out = sum(Why[i][j]*h[j] for j in range(hidden_size))
            y.append(sigmoid(sum_out + by[i]))

        # ---- Loss ----
        error = [target[i] - y[i] for i in range(vocab_size)]
        total_loss += sum(e*e for e in error)

        # ---- Backprop ----

        dy = [error[i]*sigmoid_derivative(y[i]) for i in range(vocab_size)]

        # Update Why
        for i in range(vocab_size):
            for j in range(hidden_size):
                Why[i][j] += learning_rate * dy[i] * h[j]
            by[i] += learning_rate * dy[i]

        # Hidden gradient
        dh = []
        for i in range(hidden_size):
            grad = sum(dy[j]*Why[j][i] for j in range(vocab_size))
            grad *= tanh_derivative(h[i])
            dh.append(grad)

        # Update Wxh and Whh
        for i in range(hidden_size):
            for j in range(vocab_size):
                Wxh[i][j] += learning_rate * dh[i] * x[j]
            for j in range(hidden_size):
                Whh[i][j] += learning_rate * dh[i] * h_prev[j]
            bh[i] += learning_rate * dh[i]

        h_prev = h

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("\nTraining Complete!\n")

# ------------------------
# Generate
# ------------------------

def generate(start_char, length=40):

    if start_char not in char_to_idx:
        return "Character not found."

    h_prev = [0]*hidden_size
    current_char = start_char
    result = current_char

    for _ in range(length):

        x = one_hot(char_to_idx[current_char])

        h = []
        for i in range(hidden_size):
            sum_input = sum(Wxh[i][j]*x[j] for j in range(vocab_size))
            sum_hidden = sum(Whh[i][j]*h_prev[j] for j in range(hidden_size))
            h.append(tanh(sum_input + sum_hidden + bh[i]))

        y = []
        for i in range(vocab_size):
            sum_out = sum(Why[i][j]*h[j] for j in range(hidden_size))
            y.append(sigmoid(sum_out + by[i]))

        # Sampling
        total = sum(y)
        probs = [o/total for o in y]

        r = random.random()
        cumulative = 0

        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                next_index = i
                break

        current_char = idx_to_char[next_index]
        result += current_char
        h_prev = h

    return result

while True:
    start = input("Start character: ")
    print("Generated:", generate(start))