#4. Temperature Prediction — Degrees 1, 3 and 12
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = np.arange(1, 21).reshape(-1, 1)
y = 25 + 5 * np.sin(X.ravel() / 3)

X1, X2, y1, y2 = train_test_split(
    X, y, test_size=0.3, random_state=1
)

for d in [1, 3, 12]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression())
    m.fit(X1, y1)
    print("Degree:", d)
    print("Train MSE:", mean_squared_error(y1, m.predict(X1)))
    print("Test MSE:", mean_squared_error(y2, m.predict(X2)))
    plt.plot(X, m.predict(X), label=str(d))

plt.scatter(X, y)
plt.legend()
plt.show()
