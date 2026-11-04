import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Training reviews
reviews = [
    "I love this product",
    "Excellent quality",
    "Very useful product",
    "I hate this product",
    "Very poor quality",
    "Product is terrible"
]

# 1 = Positive, 0 = Negative
y = np.array([1, 1, 1, 0, 0, 0])

# Convert text into numerical sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)

X = tokenizer.texts_to_sequences(reviews)

# Apply padding
X = pad_sequences(X, padding='post')

# Create RNN model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=8),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=100, verbose=0)

# Test with a new review
new_review = ["I love this product"]

test_input = tokenizer.texts_to_sequences(new_review)
test_input = pad_sequences(test_input,
                           maxlen=X.shape[1],
                           padding='post')

# Predict sentiment
prediction = model.predict(test_input, verbose=0)

print("Customer Review:", new_review[0])

if prediction[0][0] >= 0.5:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
