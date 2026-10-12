#3. Compare Polynomial Models of Degrees 2 and 15
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = np.sin(X.ravel())

X1, X2, y1, y2 = train_test_split(
    X, y, test_size=0.3, random_state=1
)

for d in [2, 15]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression())
    m.fit(X1, y1)
    print("Degree:", d)
    print("Train Error:", mean_squared_error(y1, m.predict(X1)))
    print("Test Error:", mean_squared_error(y2, m.predict(X2)))
    plt.plot(X, m.predict(X), label=str(d))

plt.scatter(X, y)
plt.legend()
plt.show()
