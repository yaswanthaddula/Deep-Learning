import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Training input sequences
X = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50],
    [40, 50, 60],
    [50, 60, 70]
], dtype=float)

# Expected next values
y = np.array([40, 50, 60, 70, 80], dtype=float)

# Normalize data
X = X / 100
y = y / 100

# Reshape input into RNN format
X = X.reshape(5, 3, 1)

# Create RNN model
model = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(3, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=1000, verbose=0)

# Predict next value
test_input = np.array([60, 70, 80], dtype=float)
test_input = test_input.reshape(1, 3, 1) / 100

prediction = model.predict(test_input, verbose=0)

print("Input Sequence: 60, 70, 80")
print("Predicted Next Value:", prediction[0][0] * 100)
