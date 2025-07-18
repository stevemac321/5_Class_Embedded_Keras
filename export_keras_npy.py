import tensorflow as tf
import numpy as np
from inference_strings import inference_samples_by_class
from train_keras_model import samples_by_class  # for comment block

# Flatten inference strings
inference_texts = []
for phrases in inference_samples_by_class.values():
    inference_texts.extend(phrases)

# ✅ Load vectorizer using TFSMLayer (Keras 3 compatible)
vectorizer_layer = tf.keras.layers.TFSMLayer("vectorizer_model", call_endpoint="serve")

# ✅ Reshape input to match expected shape (None, 1)
inference_tensor = tf.constant(inference_texts)[:, tf.newaxis]

# ✅ Load trained sentiment model
model = tf.keras.models.load_model("sentiment_5class_model_tfidf.keras")

# Transform and predict
X = vectorizer_layer(inference_tensor).numpy()
Y = model.predict(X)

# Confirm shape match
assert X.shape[1] == model.input_shape[1], f"Vectorizer output shape {X.shape[1]} does not match model input {model.input_shape[1]}"

# Save .npy files
np.save("cube_input.npy", X.astype(np.float32))
np.save("cube_output.npy", Y.astype(np.float32))

# Generate sentiment_test_inference.c
with open("sentiment_test_inference.c", "w") as f:
    num_features = X.shape[1]

    # Write input vectors
    for i in range(10):
        f.write(f"float review_input_{i}[5][{num_features}] = {{\n")
        for j in range(5):
            row = X[i * 5 + j]
            formatted = ", ".join(f"{x:.6f}f" for x in row)
            f.write(f"  {{{formatted}}},\n")
        f.write("};\n\n")

    # Write output vectors
    for i in range(10):
        f.write(f"float review_output_{i}[5][5] = {{\n")
        for j in range(5):
            row = Y[i * 5 + j]
            formatted = ", ".join(f"{x:.6f}f" for x in row)
            f.write(f"  {{{formatted}}},\n")
        f.write("};\n\n")

    # Write text strings
    for i in range(10):
        f.write(f'const char* review_text_{i}[5] = {{\n')
        for j in range(5):
            text = inference_texts[i * 5 + j].replace('"', '\\"')
            f.write(f'  "{text}",\n')
        f.write("};\n\n")

    # Append training strings in comment block
    f.write("/*\n")
    f.write("===========================================\n")
    f.write(" Training Strings Used to Train the Model\n")
    f.write("===========================================\n\n")
    for label in range(5):
        f.write(f"Class {label}:\n")
        for phrase in samples_by_class[label]:
            f.write(f"  - {phrase}\n")
        f.write("\n")
    f.write("*/\n")