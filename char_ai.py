import random
import math

# ------------------------
# Activation
# ------------------------

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

def one_hot(index):
    vec = [0] * vocab_size
    vec[index] = 1
    return vec

# ------------------------
# Network Parameters
# ------------------------

hidden_size = 8
learning_rate = 0.1

# Input → Hidden
w1 = [[random.uniform(-1,1) for _ in range(vocab_size)] for _ in range(hidden_size)]
b1 = [random.uniform(-1,1) for _ in range(hidden_size)]

# Hidden → Output
w2 = [[random.uniform(-1,1) for _ in range(hidden_size)] for _ in range(vocab_size)]
b2 = [random.uniform(-1,1) for _ in range(vocab_size)]

# ------------------------
# Training
# ------------------------

for epoch in range(2000):

    total_loss = 0

    for i in range(len(text)-1):

        x_char = text[i]
        y_char = text[i+1]

        x = one_hot(char_to_idx[x_char])
        target = one_hot(char_to_idx[y_char])

        # Forward pass
        hidden = []
        for h in range(hidden_size):
            z = sum(x[j]*w1[h][j] for j in range(vocab_size)) + b1[h]
            hidden.append(sigmoid(z))

        output = []
        for o in range(vocab_size):
            z = sum(hidden[h]*w2[o][h] for h in range(hidden_size)) + b2[o]
            output.append(sigmoid(z))

        # Loss
        error = [target[o] - output[o] for o in range(vocab_size)]
        total_loss += sum(e*e for e in error)

        # Backprop output layer
        d_output = [error[o] * sigmoid_derivative(output[o]) for o in range(vocab_size)]

        for o in range(vocab_size):
            for h in range(hidden_size):
                w2[o][h] += learning_rate * d_output[o] * hidden[h]
            b2[o] += learning_rate * d_output[o]

        # Backprop hidden layer
        for h in range(hidden_size):
            grad = sum(d_output[o]*w2[o][h] for o in range(vocab_size))
            grad *= sigmoid_derivative(hidden[h])

            for j in range(vocab_size):
                w1[h][j] += learning_rate * grad * x[j]
            b1[h] += learning_rate * grad

    if epoch % 400 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("\nTraining Complete!\n")

# ------------------------
# Generate Text (Creative Mode)
# ------------------------

def generate(start_char, length=30):

    if start_char not in char_to_idx:
        return "Character not in vocabulary."

    current_char = start_char
    result = current_char

    for _ in range(length):

        x = one_hot(char_to_idx[current_char])

        hidden = []
        for h in range(hidden_size):
            z = sum(x[j]*w1[h][j] for j in range(vocab_size)) + b1[h]
            hidden.append(sigmoid(z))

        output = []
        for o in range(vocab_size):
            z = sum(hidden[h]*w2[o][h] for h in range(hidden_size)) + b2[o]
            output.append(sigmoid(z))

        # -------- Creative Sampling --------

        total = sum(output)
        probs = [o/total for o in output]

        r = random.random()
        cumulative = 0

        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                next_index = i
                break

        current_char = idx_to_char[next_index]
        result += current_char

    return result


while True:
    start = input("Start character: ")
    print("Generated:", generate(start))