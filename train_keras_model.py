import tensorflow as tf
import numpy as np

# Curated training strings by class
samples_by_class = {
    0: [
        "I hated every second of this.",
        "This was a complete disaster.",
        "Absolutely awful from start to finish.",
        "I couldn’t stand watching it.",
        "The worst movie I’ve ever seen.",
        "Painfully bad and poorly made.",
        "This was one of the worst films I’ve ever seen.",
        "Unbearably boring and stupid.",
        "A total failure in every way.",
        "This film was garbage."
    ],
    1: [
        "Not a good movie at all.",
        "Mostly disappointing and dull.",
        "Had potential but failed.",
        "The plot was terrible and the performances were awful.",
        "This movie was full of problems and completely unenjoyable.",
        "This movie was boring and poorly made.",
        "Couldn’t get into it.",
        "Just didn’t work for me.",
        "Below average and boring.",
        "Not worth watching."
    ],
    2: [
        "It was fine, nothing special.",
        "An average experience.",
        "Neither good nor bad.",
        "Some parts were okay.",
        "A typical movie night.",
        "Decent but not memorable.",
        "Mildly entertaining.",
        "Just a typical movie, nothing stood out.",
        "Had good and bad moments, overall average.",
        "Just okay overall."
    ],
    3: [
        "I liked it a lot.",
        "A solid and enjoyable film.",
        "Well made and entertaining.",
        "Good performances and story.",
        "This movie exceeded my expectations.",
        "Would recommend to others.",
        "Nicely done and satisfying.",
        "A good watch.",
        "Enjoyable from start to end.",
        "A thoroughly satisfying and well-crafted movie experience."
    ],
    4: [
        "Absolutely loved it!",
        "This was a brilliant and beautifully told story.",
        "Brilliant and unforgettable.",
        "One of the best movies ever.",
        "Emotionally powerful and beautiful.",
        "Stunning visuals and acting.",
        "Perfect in every way.",
        "Exceeded all expectations.",
        "An absolutely phenomenal film — one of the very best.",
        "I loved it so much I’ll definitely rewatch it."
    ]
}

# Flatten samples and labels
texts = []
labels = []
for label, phrases in samples_by_class.items():
    texts.extend(phrases)
    labels.extend([label] * len(phrases))

# Convert to tensors
texts_tensor = tf.constant(texts)
labels_tensor = tf.constant(labels)

# Text vectorization layer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=1000,
    output_mode='tf-idf'
)
vectorizer.adapt(texts_tensor)

# Transform inputs
X = vectorizer(texts_tensor)
y = labels_tensor

# Build model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(5, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=100, verbose=1)

# Save model for X-CUBE-AI
model.save("sentiment_5class_model_tfidf.keras")

# ✅ Save vectorizer using SavedModel format (not .keras)
inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
outputs = vectorizer(inputs)
vectorizer_model = tf.keras.Model(inputs, outputs)
#vectorizer_model.save("vectorizer_model", save_format="tf")
vectorizer_model.export("vectorizer_model")

# Archive training strings
with open("training_strings.txt", "w") as f:
    for label in range(5):
        f.write(f"Class {label}:\n")
        for phrase in samples_by_class[label]:
            f.write(f"  - {phrase}\n")
        f.write("\n")