#2. Normal Distribution — Examination Marks
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean = 70
sd = 10

prob = norm.cdf(80, mean, sd) - norm.cdf(60, mean, sd)
print("Probability between 60 and 80:", round(prob, 4))

x = np.linspace(40, 100, 200)
y = norm.pdf(x, mean, sd)

plt.plot(x, y)
plt.fill_between(x, y, where=(x >= 60) & (x <= 80), alpha=0.5)
plt.title("Normal Distribution")
plt.show()
