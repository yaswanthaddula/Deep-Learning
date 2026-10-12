#1. Entropy of Weather Conditions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

P = np.array([0.6, 0.3, 0.1])

H = entropy(P, base=2)
print("Entropy:", round(H, 4))

plt.bar(["Sunny", "Rainy", "Cloudy"], P)
plt.ylabel("Probability")
plt.title("Weather Conditions")
plt.show()
