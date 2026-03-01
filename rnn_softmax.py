import random
import math

# ------------------------
# Activation
# ------------------------

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - x*x

def softmax(vec):
    max_val = max(vec)  # stability
    exps = [math.exp(v - max_val) for v in vec]
    total = sum(exps)
    return [e/total for e in exps]

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

Wxh = [[random.uniform(-0.5,0.5) for _ in range(vocab_size)] for _ in range(hidden_size)]
Whh = [[random.uniform(-0.5,0.5) for _ in range(hidden_size)] for _ in range(hidden_size)]
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
        target_index = char_to_idx[text[t+1]]

        # ---- Forward ----

        h = []
        for i in range(hidden_size):
            sum_input = sum(Wxh[i][j]*x[j] for j in range(vocab_size))
            sum_hidden = sum(Whh[i][j]*h_prev[j] for j in range(hidden_size))
            h.append(tanh(sum_input + sum_hidden + bh[i]))

        raw_output = []
        for i in range(vocab_size):
            sum_out = sum(Why[i][j]*h[j] for j in range(hidden_size))
            raw_output.append(sum_out + by[i])

        y = softmax(raw_output)

        # ---- Cross-Entropy Loss ----
        loss = -math.log(y[target_index] + 1e-9)
        total_loss += loss

        # ---- Backprop ----

        dy = y[:]
        dy[target_index] -= 1  # derivative of cross-entropy softmax

        # Update Why
        for i in range(vocab_size):
            for j in range(hidden_size):
                Why[i][j] -= learning_rate * dy[i] * h[j]
            by[i] -= learning_rate * dy[i]

        # Hidden gradient
        dh = []
        for i in range(hidden_size):
            grad = sum(dy[j]*Why[j][i] for j in range(vocab_size))
            grad *= tanh_derivative(h[i])
            dh.append(grad)

        # Update Wxh and Whh
        for i in range(hidden_size):
            for j in range(vocab_size):
                Wxh[i][j] -= learning_rate * dh[i] * x[j]
            for j in range(hidden_size):
                Whh[i][j] -= learning_rate * dh[i] * h_prev[j]
            bh[i] -= learning_rate * dh[i]

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

        raw_output = []
        for i in range(vocab_size):
            sum_out = sum(Why[i][j]*h[j] for j in range(hidden_size))
            raw_output.append(sum_out + by[i])

        y = softmax(raw_output)

        # Sampling
        r = random.random()
        cumulative = 0
        for i, p in enumerate(y):
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