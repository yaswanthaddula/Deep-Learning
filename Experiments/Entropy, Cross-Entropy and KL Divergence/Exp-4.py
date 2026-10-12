#4. Compare Entropy of Two Predictions
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

P = np.array([0.9, 0.1])
Q = np.array([0.5, 0.5])

H1 = entropy(P, base=2)
H2 = entropy(Q, base=2)

print("Entropy of P:", round(H1, 4))
print("Entropy of Q:", round(H2, 4))
print("Greater Uncertainty: Q")

plt.bar(["P", "Q"], [H1, H2])
plt.ylabel("Entropy (bits)")
plt.title("Prediction Uncertainty")
plt.show()
