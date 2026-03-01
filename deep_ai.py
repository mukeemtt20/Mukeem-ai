import random
import math

# -----------------------
# Activation Functions
# -----------------------

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -----------------------
# Initialize Weights
# -----------------------

# Input → Hidden1 (2x4)
w1 = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(4)]
b1 = [random.uniform(-1, 1) for _ in range(4)]

# Hidden1 → Hidden2 (4x3)
w2 = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
b2 = [random.uniform(-1, 1) for _ in range(3)]

# Hidden2 → Output (3)
w3 = [random.uniform(-1, 1) for _ in range(3)]
b3 = random.uniform(-1, 1)

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

for epoch in range(3000):

    total_loss = 0

    for inputs, target in data:
        x1, x2 = inputs

        # ---- Forward Pass ----

        # Hidden Layer 1
        h1 = []
        for i in range(4):
            z = x1*w1[i][0] + x2*w1[i][1] + b1[i]
            h1.append(sigmoid(z))

        # Hidden Layer 2
        h2 = []
        for i in range(3):
            z = sum(h1[j]*w2[i][j] for j in range(4)) + b2[i]
            h2.append(sigmoid(z))

        # Output
        z_out = sum(h2[i]*w3[i] for i in range(3)) + b3
        output = sigmoid(z_out)

        # ---- Loss ----
        error = target - output
        total_loss += error**2

        # ---- Backprop ----

        d_output = error * sigmoid_derivative(output)

        # Update Hidden2 → Output
        for i in range(3):
            w3[i] += learning_rate * d_output * h2[i]
        b3 += learning_rate * d_output

        # Hidden2 gradients
        d_h2 = []
        for i in range(3):
            grad = d_output * w3[i] * sigmoid_derivative(h2[i])
            d_h2.append(grad)

        # Update Hidden1 → Hidden2
        for i in range(3):
            for j in range(4):
                w2[i][j] += learning_rate * d_h2[i] * h1[j]
            b2[i] += learning_rate * d_h2[i]

        # Hidden1 gradients
        for i in range(4):
            grad = sum(d_h2[j] * w2[j][i] for j in range(3))
            grad *= sigmoid_derivative(h1[i])

            w1[i][0] += learning_rate * grad * x1
            w1[i][1] += learning_rate * grad * x2
            b1[i] += learning_rate * grad

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------
# Testing
# -----------------------

print("\nTraining Complete!\n")

test_input = [4, 5]

# Forward pass only
h1 = []
for i in range(4):
    z = test_input[0]*w1[i][0] + test_input[1]*w1[i][1] + b1[i]
    h1.append(sigmoid(z))

h2 = []
for i in range(3):
    z = sum(h1[j]*w2[i][j] for j in range(4)) + b2[i]
    h2.append(sigmoid(z))

z_out = sum(h2[i]*w3[i] for i in range(3)) + b3
prediction = sigmoid(z_out)

print(f"Test Input: {test_input}")
print(f"Prediction: {prediction:.4f}")