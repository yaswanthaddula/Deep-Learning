#4. Calculate Eigenvalues and Eigenvectors and Plot Arrows
import numpy as np
import matplotlib.pyplot as plt

C = np.array([[3, 1],
              [1, 3]])

values, vectors = np.linalg.eig(C)
print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

for i in range(2):
    plt.quiver(0, 0, vectors[0, i], vectors[1, i],
               angles="xy", scale_units="xy", scale=1)

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.title("Eigenvectors")
plt.show()
