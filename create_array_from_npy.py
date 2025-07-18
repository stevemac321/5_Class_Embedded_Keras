import numpy as np

# Load vectors
X = np.load("cube_input.npy")
Y = np.load("cube_output.npy")

# Validate shape
assert X.shape == (50, 1000), "Expected 50 samples with 1000 features"
assert Y.shape == (50, 5), "Expected 50 samples with 5 output scores"

# 50 review strings
review_strings = [
    "I hated every second of this.",
    "This was a complete disaster.",
    "Absolutely awful from start to finish.",
    "I couldn’t stand watching it.",
    "The worst movie I’ve ever seen.",
    "Painfully bad and poorly made.",
    "This was one of the worst films I’ve ever seen.",
    "Unbearably boring and stupid.",
    "A total failure in every way.",
    "This film was garbage.",
    "Not a good movie at all.",
    "Mostly disappointing and dull.",
    "Had potential but failed.",
    "The plot was terrible and the performances were awful.",
    "This movie was full of problems and completely unenjoyable.",
    "This movie was boring and poorly made.",
    "Couldn’t get into it.",
    "Just didn’t work for me.",
    "Below average and boring.",
    "Not worth watching.",
    "It was fine, nothing special.",
    "An average experience.",
    "Neither good nor bad.",
    "Some parts were okay.",
    "A typical movie night.",
    "Decent but not memorable.",
    "Mildly entertaining.",
    "Just a typical movie, nothing stood out.",
    "Had good and bad moments, overall average.",
    "Just okay overall.",
    "I liked it a lot.",
    "A solid and enjoyable film.",
    "Well made and entertaining.",
    "Good performances and story.",
    "This movie exceeded my expectations.",
    "Would recommend to others.",
    "Nicely done and satisfying.",
    "A good watch.",
    "Enjoyable from start to end.",
    "A thoroughly satisfying and well-crafted movie experience.",
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

with open("sentiment_test_vectors.c", "w") as f:
    f.write("// Sentiment test vectors for 5-class model\n\n")

    for group in range(10):
        start = group * 5
        end = start + 5
        X_group = X[start:end]
        Y_group = Y[start:end]
        texts_group = review_strings[start:end]

        # Input vectors
        f.write(f"float review_input_{group}[5][1000] = {{\n")
        for row in X_group:
            f.write("  {\n")
            for j in range(0, 1000, 100):
                line = ", ".join(f"{row[k]:.6f}f" for k in range(j, j + 100))
                f.write(f"    {line},\n")
            f.write("  },\n")
        f.write("};\n\n")

        # Output vectors
        f.write(f"const float review_output_{group}[5][5] = {{\n")
        for row in Y_group:
            line = ", ".join(f"{val:.6f}f" for val in row)
            f.write(f"  {{ {line} }},\n")
        f.write("};\n\n")

        # Text strings
        f.write(f"const char* review_text_{group}[5] = {{\n")
        for text in texts_group:
            escaped = text.replace('"', '\\"')
            f.write(f'  "{escaped}",\n')
        f.write("};\n\n")