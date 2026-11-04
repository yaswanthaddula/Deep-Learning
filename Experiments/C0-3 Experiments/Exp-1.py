import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Input study hours
X = np.array([2, 3, 4, 5], dtype=float)

# Normalize data
X = X / 10

# Reshape input into RNN format
X = X.reshape(1, 4, 1)

# Create RNN model
model = Sequential([
    SimpleRNN(32, activation='tanh', input_shape=(4, 1)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Display model summary
model.summary()

# Predict output
prediction = model.predict(X, verbose=0)

print("Input Study Hours: 2, 3, 4, 5")
print("Predicted Output:", prediction[0][0] * 10)
