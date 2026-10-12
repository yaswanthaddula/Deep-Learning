#2. Covariance Matrix of Temperature and Humidity
import numpy as np
import matplotlib.pyplot as plt

temp = [22, 25, 28, 30, 32]
humidity = [80, 75, 68, 60, 55]

C = np.cov(temp, humidity)
print("Covariance Matrix:\n", C)

plt.scatter(temp, humidity)
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Temperature vs Humidity")
plt.show()
