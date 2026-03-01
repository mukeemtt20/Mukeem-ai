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
# Initialize Weights
# -----------------------

# Input → Hidden (2x3 = 6 weights)
w = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(3)]
b_hidden = [random.uniform(-1, 1) for _ in range(3)]

# Hidden → Output (3 weights)
v = [random.uniform(-1, 1) for _ in range(3)]
b_output = random.uniform(-1, 1)

learning_rate = 0.1

# -----------------------
# Training Data
# -----------------------
data = [
    ([2, 3], 0),
    ([5, 6], 1),
    ([1, 2], 0),
    ([6, 7], 1),
    ([3, 4], 0),
    ([7, 8], 1)
]

# -----------------------
# Training Loop
# -----------------------

for epoch in range(2000):

    total_loss = 0

    for inputs, target in data:
        x1, x2 = inputs

        # ---- Forward Pass ----

        hidden_outputs = []
        for i in range(3):
            z = x1 * w[i][0] + x2 * w[i][1] + b_hidden[i]
            hidden_outputs.append(sigmoid(z))

        z_output = sum(hidden_outputs[i] * v[i] for i in range(3)) + b_output
        output = sigmoid(z_output)

        # ---- Loss ----
        error = target - output
        total_loss += error ** 2

        # ---- Backprop ----

        d_output = error * sigmoid_derivative(output)

        # Update hidden → output weights
        for i in range(3):
            v[i] += learning_rate * d_output * hidden_outputs[i]

        b_output += learning_rate * d_output

        # Hidden layer gradients
        for i in range(3):
            d_hidden = d_output * v[i] * sigmoid_derivative(hidden_outputs[i])

            w[i][0] += learning_rate * d_hidden * x1
            w[i][1] += learning_rate * d_hidden * x2
            b_hidden[i] += learning_rate * d_hidden

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------
# Testing
# -----------------------

print("\nTraining Complete!\n")

test_input = [4, 5]

hidden_test = []
for i in range(3):
    z = test_input[0] * w[i][0] + test_input[1] * w[i][1] + b_hidden[i]
    hidden_test.append(sigmoid(z))

z_out = sum(hidden_test[i] * v[i] for i in range(3)) + b_output
prediction = sigmoid(z_out)

print(f"Test Input: {test_input}")
print(f"Prediction: {prediction:.4f}")