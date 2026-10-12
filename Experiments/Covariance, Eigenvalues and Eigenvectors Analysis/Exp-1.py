#1. Covariance Between Study Hours and Marks
import numpy as np
import matplotlib.pyplot as plt

hours = [2, 4, 6, 8, 10]
marks = [35, 45, 60, 75, 85]

cov = np.cov(hours, marks)[0, 1]
print("Sample Covariance:", cov)

plt.scatter(hours, marks)
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.title("Study Hours vs Marks")
plt.show()
