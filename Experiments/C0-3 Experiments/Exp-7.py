import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Training sentences
sentences = [
    "I love machine learning",
    "I love deep learning",
    "I love artificial intelligence",
    "I like machine learning",
    "I like deep learning"
]

# Convert words into numbers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Create input and output sequences
X = []
y = []

for sentence in sentences:
    sequence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(sequence)):
        X.append(sequence[:i])
        y.append(sequence[i])

# Apply padding
X = pad_sequences(X, padding='pre')
y = np.array(y)

# Create RNN model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,
              output_dim=10),
    SimpleRNN(32),
    Dense(len(tokenizer.word_index) + 1,
          activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=500, verbose=0)

# Predict next word
input_text = "I love"

test_input = tokenizer.texts_to_sequences([input_text])
test_input = pad_sequences(test_input,
                           maxlen=X.shape[1],
                           padding='pre')

prediction = model.predict(test_input, verbose=0)

predicted_index = np.argmax(prediction[0])
predicted_word = tokenizer.index_word[predicted_index]

print("Input Phrase:", input_text)
print("Predicted Next Word:", predicted_word)
