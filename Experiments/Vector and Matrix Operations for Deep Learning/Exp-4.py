#4. Update Weights Using Gradient Descent
import numpy as np

W = np.array([0.7, 0.5, 0.9])
G = np.array([0.2, 0.1, 0.3])
learning_rate = 0.1

W_new = W - learning_rate * G

print("Updated Weights:", W_new)
