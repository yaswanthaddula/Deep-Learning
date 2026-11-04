import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Training text
text = "machine learning is interesting"

# Create character vocabulary
chars = sorted(set(text))

# Convert characters into numbers
char_to_num = {c: i for i, c in enumerate(chars)}
num_to_char = {i: c for i, c in enumerate(chars)}

# Create input and output sequences
X = []
y = []

for i in range(len(text) - 4):
    X.append([char_to_num[c] for c in text[i:i+4]])
    y.append(char_to_num[text[i+4]])

X = np.array(X)
y = np.array(y)

# One-hot encoding
X = np.eye(len(chars))[X]
y = np.eye(len(chars))[y]

# Create RNN model
model = Sequential([
    SimpleRNN(50, input_shape=(4, len(chars))),
    Dense(len(chars), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=300, verbose=0)

# Generate characters
start = "lear"
generated = start

for i in range(10):
    test_input = np.array([
        [char_to_num[c] for c in generated[-4:]]
    ])

    test_input = np.eye(len(chars))[test_input]

    prediction = model.predict(test_input, verbose=0)

    next_char = num_to_char[np.argmax(prediction[0])]

    generated += next_char

print("Starting Sequence:", start)
print("Generated Text:", generated)
