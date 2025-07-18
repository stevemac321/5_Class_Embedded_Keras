

```markdown
# 5_Class_Embedded_Keras

This repository is **not a complete STM32 project**, but a **walkthrough** showing how to build, validate, and deploy a 5-class sentiment classifier on embedded hardware using Keras, TF-IDF vectorization, and X-CUBE-AI.

It upgrades a previous 3-class naive model with finer granularity and full parity between desktop and embedded inference.

---

## 🎯 Overview

- **Sentiment Classes**:
- 0: Extreme Negative
- 1: Strong Negative
- 2: Moderate Negative
- 3: Positive
- 4: Strong Positive




- **Model Input**: TF-IDF vector (133 features)  
- **Model Output**: Softmax over 5 classes  
- **Deployment Target**: STM32 (tested on STM32H723ZG)  
- **Validation**: 100% match between Python and embedded predictions

---

## 🧠 Walkthrough Steps

### 1️⃣ Train the Model

Use `train_keras_model.py` to train a 5-class sentiment model from `training_strings.txt`:

```bash
python train_keras_model.py
```

This produces:
- `sentiment_model_v2.keras` — the trained model
- `vectorizer_model.keras` — the saved TF-IDF vectorizer
- `tfidf_vectorizer.pkl` — optional pickled vectorizer

---

### 2️⃣ Export Inference Vectors

Use `export_keras_npy.py` to run inference and save input/output vectors:

```bash
python export_keras_npy.py
```

This produces:
- `cube_input.npy` — TF-IDF vectors
- `cube_output_new.npy` — softmax predictions
- `inference_strings.py` — optional string mapping

---

### 3️⃣ Convert to Embedded Format

Use `create_array_from_npy.py` to convert `.npy` files into C arrays:

```bash
python create_array_from_npy.py
```

This generates:
- `sentiment_test_vectors.c` — input/output arrays
- `sentiment_test_vectors.h` — declarations

---

### 4️⃣ Import Model into X-CUBE-AI

- Open STM32CubeMX  
- Import `sentiment_model_v2.keras`  
- Verify input shape `(133,)`, output shape `(5,)`  
- Click **Analyze** → **Generate**

---

### 5️⃣ Integrate and Run on STM32

Use the following embedded files:

- `main.c` — inference loop and logging  
- `app_x-cube-ai.c` — AI middleware integration  
- `sentiment_test_vectors.c/h` — test cases and expected outputs

Flash to your STM32 target and run inference. Predictions will be printed over UART or console.

---

## 📁 File Descriptions

| File | Purpose |
|------|--------|
| `main.c` | Embedded inference loop and logging |
| `app_x-cube-ai.c` | AI middleware glue code |
| `sentiment_model_v2.keras` | Trained 5-class Keras model |
| `vectorizer_model.keras` | Saved TF-IDF vectorizer |
| `tfidf_vectorizer.pkl` | Pickled vectorizer (optional) |
| `training_strings.txt` | Raw text samples used for training |
| `cube_input.npy` | TF-IDF input vectors |
| `cube_output_new.npy` | Softmax predictions |
| `sentiment_test_vectors.c/h` | Embedded test vectors |
| `train_keras_model.py` | Script to train and export model |
| `export_keras_npy.py` | Script to run inference and save `.npy` files |
| `create_array_from_npy.py` | Converts `.npy` to C arrays |
| `inference_strings.py` | Optional string mapping for logging |

---

## 📹 YouTube Demo Coming Soon

A full walkthrough video will be released soon, showing:

- Model training  
- Python validation  
- Embedded deployment  
- Live inference on STM32

Stay tuned!

---

## 🧪 Validation Snapshot

- 45 samples tested  
- 100% match between expected and actual predictions  
- Confidence scores aligned  
- Embedded inference verified against Python model

---

## 🛠️ Notes

- This repo is a walkthrough, not a full STM32CubeIDE project  
- You’ll need to integrate the files into your own STM32 project manually  
- Tested with X-CUBE-AI and STM32H723ZG, but portable to other targets

