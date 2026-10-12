#5. Learning Curves — Linear and Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.linspace(-3, 3, 40).reshape(-1, 1)
y = X.ravel()**2 + np.random.randn(40) * 0.2

models = [
    LinearRegression(),
    make_pipeline(PolynomialFeatures(10), LinearRegression())
]

for m in models:
    sizes, train, test = learning_curve(
        m, X, y, cv=3, scoring="neg_mean_squared_error"
    )
    plt.plot(sizes, -train.mean(axis=1), marker="o")
    plt.plot(sizes, -test.mean(axis=1), marker="s")

plt.xlabel("Training Samples")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curves")
plt.show()
