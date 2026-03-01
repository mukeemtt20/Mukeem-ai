import random
import math

# -----------------------
# Activation
# -----------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -----------------------
# Dataset
# -----------------------

dataset = [
    ("i love this", 1),
    ("this is amazing", 1),
    ("i feel great", 1),
    ("i hate this", 0),
    ("this is bad", 0),
    ("i feel terrible", 0)
]

# -----------------------
# Build Vocabulary
# -----------------------

vocab = set()

for sentence, _ in dataset:
    for word in sentence.split():
        vocab.add(word)

vocab = list(vocab)

def encode(sentence):
    words = sentence.split()
    return [1 if word in words else 0 for word in vocab]

# -----------------------
# Initialize Weights
# -----------------------

input_size = len(vocab)
hidden_size = 6

# Input → Hidden
w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

# Hidden → Output
w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
b2 = random.uniform(-1, 1)

learning_rate = 0.1

# -----------------------
# Training
# -----------------------

for epoch in range(2000):

    total_loss = 0

    for sentence, target in dataset:

        x = encode(sentence)

        # Forward pass

        hidden = []
        for i in range(hidden_size):
            z = sum(x[j] * w1[i][j] for j in range(input_size)) + b1[i]
            hidden.append(sigmoid(z))

        z_out = sum(hidden[i] * w2[i] for i in range(hidden_size)) + b2
        output = sigmoid(z_out)

        error = target - output
        total_loss += error ** 2

        # Backprop

        d_output = error * sigmoid_derivative(output)

        # Update hidden → output
        for i in range(hidden_size):
            w2[i] += learning_rate * d_output * hidden[i]

        b2 += learning_rate * d_output

        # Hidden gradients
        for i in range(hidden_size):
            d_hidden = d_output * w2[i] * sigmoid_derivative(hidden[i])

            for j in range(input_size):
                w1[i][j] += learning_rate * d_hidden * x[j]

            b1[i] += learning_rate * d_hidden

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------
# Testing
# -----------------------

print("\nTraining Complete!\n")

while True:
    test = input("Enter sentence: ")
    x = encode(test)

    hidden = []
    for i in range(hidden_size):
        z = sum(x[j] * w1[i][j] for j in range(input_size)) + b1[i]
        hidden.append(sigmoid(z))

    z_out = sum(hidden[i] * w2[i] for i in range(hidden_size)) + b2
    prediction = sigmoid(z_out)

    print(f"Prediction (0=Negative, 1=Positive): {prediction:.4f}")