#3. KL Divergence Between Two Language Models
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.4, 0.4, 0.2])

print("KL Divergence:", round(entropy(P, Q), 4))

x = np.arange(3)
plt.bar(x - 0.2, P, 0.4, label="P")
plt.bar(x + 0.2, Q, 0.4, label="Q")
plt.xticks(x, ["Word 1", "Word 2", "Word 3"])
plt.legend()
plt.show()
