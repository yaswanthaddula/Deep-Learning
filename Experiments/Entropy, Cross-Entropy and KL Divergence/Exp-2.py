#2. Cross-Entropy Loss for Image Classification
import numpy as np
import matplotlib.pyplot as plt

Y = np.array([1, 0, 0])
P = np.array([0.7, 0.2, 0.1])

loss = -np.sum(Y * np.log(P))
print("Cross-Entropy Loss:", round(loss, 4))

x = np.arange(3)
plt.bar(x - 0.2, Y, 0.4, label="Actual")
plt.bar(x + 0.2, P, 0.4, label="Predicted")
plt.xticks(x, ["Cat", "Dog", "Bird"])
plt.legend()
plt.show()
