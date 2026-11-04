import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Daily electricity consumption
consumption = np.array([
    100, 115, 120, 130, 145, 150,
    165, 170, 185, 190, 205, 215
], dtype=float)

# Normalize data
data = consumption / 250

# Create input and output sequences
X = []
y = []

for i in range(len(data) - 3):
    X.append(data[i:i+3])
    y.append(data[i+3])

X = np.array(X)
y = np.array(y)

# Reshape input into RNN format
X = X.reshape(9, 3, 1)

# Create RNN model
model = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(3, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=1000, verbose=0)

# Predict Day 13 consumption
test_input = np.array([190, 205, 215], dtype=float)
test_input = test_input.reshape(1, 3, 1) / 250

prediction = model.predict(test_input, verbose=0)

print("Input Consumption: 190, 205, 215")
print("Predicted Day 13 Consumption:", prediction[0][0] * 250)
