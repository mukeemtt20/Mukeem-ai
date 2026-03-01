import random
import math

# -----------------------
# Sigmoid Activation
# -----------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# -----------------------
# Initialize weights
# -----------------------
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
bias = random.uniform(-1, 1)

learning_rate = 0.1

# -----------------------
# Training Data
# -----------------------
# [hours_study, sleep_hours], result
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
for epoch in range(1000):

    total_loss = 0

    for inputs, target in data:
        x1, x2 = inputs

        # Forward pass
        weighted_sum = (x1 * w1) + (x2 * w2) + bias
        output = sigmoid(weighted_sum)

        # Calculate error
        error = target - output
        total_loss += error ** 2

        # Backpropagation
        d_output = error * sigmoid_derivative(output)

        # Update weights
        w1 += learning_rate * d_output * x1
        w2 += learning_rate * d_output * x2
        bias += learning_rate * d_output

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# -----------------------
# Testing
# -----------------------
print("\nTraining Complete!\n")

test_input = [4, 5]
weighted_sum = (test_input[0] * w1) + (test_input[1] * w2) + bias
prediction = sigmoid(weighted_sum)

print(f"Test Input: {test_input}")
print(f"Prediction (0-1): {prediction:.4f}")