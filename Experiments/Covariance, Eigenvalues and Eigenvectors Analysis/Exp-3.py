#3. Calculate Eigenvalues and Display a Bar Chart
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[4, 2],
              [2, 4]])

values = np.linalg.eigvals(A)
print("Eigenvalues:", values)

plt.bar(["Eigenvalue 1", "Eigenvalue 2"], values)
plt.title("Eigenvalues")
plt.ylabel("Value")
plt.show()
