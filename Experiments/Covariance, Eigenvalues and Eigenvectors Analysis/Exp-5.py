#5. Identify the Principal Direction Using Eigenvalues and Eigenvectors
import numpy as np
import matplotlib.pyplot as plt

C = np.array([[5, 2],
              [2, 2]])

values, vectors = np.linalg.eig(C)

print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

i = np.argmax(values)
print("Largest Eigenvalue:", values[i])
print("Principal Eigenvector:", vectors[:, i])

for j in range(2):
    plt.quiver(0, 0, vectors[0, j], vectors[1, j],
               angles="xy", scale_units="xy", scale=1,
               label=f"Eigenvector {j+1}")

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.title("Principal Direction")
plt.show()
