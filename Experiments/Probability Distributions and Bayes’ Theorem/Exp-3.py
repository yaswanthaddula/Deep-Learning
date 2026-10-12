#3. Poisson Distribution — Customer Arrivals
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

x = np.arange(0, 13)
p = poisson.pmf(x, 4)

print("Probability of exactly 6 customers:",
      round(poisson.pmf(6, 4), 4))

plt.bar(x, p)
plt.xlabel("Number of Customers")
plt.ylabel("Probability")
plt.title("Poisson Distribution")
plt.show()
