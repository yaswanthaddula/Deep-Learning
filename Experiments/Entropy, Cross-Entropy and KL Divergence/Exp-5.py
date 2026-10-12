#5. Cross-Entropy and KL Divergence for Four Categories
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

Y = np.array([0, 1, 0, 0])
P = np.array([0.1, 0.6, 0.2, 0.1])

CE = -np.sum(Y * np.log(P))
KL = entropy(Y, P)

print("Cross-Entropy Loss:", round(CE, 4))
print("KL Divergence:", round(KL, 4))

x = np.arange(4)
plt.bar(x - 0.2, Y, 0.4, label="Actual")
plt.bar(x + 0.2, P, 0.4, label="Predicted")
plt.xticks(x, ["A", "B", "C", "D"])
plt.legend()
plt.show()
