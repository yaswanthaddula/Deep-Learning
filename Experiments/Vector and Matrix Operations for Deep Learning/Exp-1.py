#1. Flatten Image Matrix and Calculate Sum
import numpy as np

A = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

V = A.flatten()

print("Flattened Vector:", V)
print("Sum of Pixels:", np.sum(A))
