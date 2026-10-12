#1. Compare Polynomial Regression Models of Degrees 1, 2, and 10
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(1)
X = np.linspace(1, 10, 20).reshape(-1, 1)
y = X.ravel()**2 + np.random.randn(20) * 5

x = np.linspace(1, 10, 100).reshape(-1, 1)

plt.scatter(X, y, label="Actual Data")

for d in [1, 2, 10]:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X, y)
    plt.plot(x, model.predict(x), label=f"Degree {d}")

plt.title("Polynomial Regression")
plt.xlabel("House Area")
plt.ylabel("House Price")
plt.legend()
plt.show()
