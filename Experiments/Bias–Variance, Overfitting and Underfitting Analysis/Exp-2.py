#2. Linear Regression on Nonlinear Student Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.arange(1, 21).reshape(-1, 1)
y = X.ravel()**2 + np.random.randn(20) * 5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Training MSE:",
      mean_squared_error(y_train, model.predict(X_train)))
print("Testing MSE:",
      mean_squared_error(y_test, model.predict(X_test)))

plt.scatter(X, y, label="Actual Marks")
plt.scatter(X, model.predict(X), label="Predicted Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.show()
