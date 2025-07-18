
# 5_Class_Embedded_Keras
## 🧠 Custom Labeling for Domain-Specific Inference

This project trains on user-defined sentiment strings, not public datasets. It demonstrates a reproducible workflow for creating labeled data, training a multi-class model, validating predictions, and deploying to embedded systems using X-CUBE-AI. The approach applies broadly to any scenario where structured labels are used — such as anomaly detection, fault classification, or domain-specific categorization — and supports full parity between desktop inference and embedded deployment.

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
- **Deployment Target**: STM32 (tested on STM32H723ZG and STM32F401RE)  
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
- 'sentiment_test_vectors.h' and 'sentiment_test_vectors.h'

---
(step 3 has been deleted)

### 4️⃣ Create and Configure STM32 Project

#### ✅ STM32CubeMX Setup

1. Create a new STM32 project (tested on STM32F4 and STM32H7)
2. Enable **X-CUBE-AI middleware**, core, and application template
3. Add (+) a network and import `sentiment_model_v2.keras`
4. Validate input/output shapes using `.npy` files
5. Increase RAM and stack to `0x1000` (works for both boards)
6. Generate code

#### ✅ STM32CubeIDE Integration

7. Build once to trigger full codegen  
8. Add `syscalls.c` to fix `_lwrite`, `_read`, etc.  
9. Go to **Project → Properties → MCU Settings** and enable **Floating Point `printf`**  
10. Copy `_write()` from the cloned `main.c` and match `huartX` to your UART handle  
11. Include `sentiment_test_vectors.h` in `main.c`  
12. Add `.h` to `Core/Inc`, `.c` to `Core/Src`  
13. Comment out the default call to `MX_X_CUBE_AI_Process()` in the `while(1)` loop  
14. Include `sentiment_test_vectors.h` in `app_x-cube-ai.c`  
15. Add user variables:
    ```c
    extern const float* user_inputs;
    extern const float* expected_output;
    extern const char* text;
    extern int global_index;
    ```
16. Replace `acquire_and_process_data()` and `post_process()` with the cloned versions  
17. Replace the user code in `MX_X_CUBE_AI_Process()` with the cloned version  
18. If you regenerate code, reapply:
    - `syscalls.c`  
    - `_write()` override  
    - UART handle  
    - Inference logic

---

## ⚠️ Floating-Point Printing in `printf()`

By default, embedded `printf()` does **not support floating-point formatting** (e.g. `%.2f`) unless explicitly enabled.

### ✅ STM32CubeIDE (GCC / newlib-nano)

1. Go to **Project → Properties → C/C++ Build → Settings**  
2. Under **Tool Settings → MCU GCC Linker → Miscellaneous**, add:
   ```
   -u _printf_float
   ```
3. Clean and rebuild the project

This ensures confidence scores and softmax outputs are printed correctly during inference.

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
| `export_keras_npy.py` | Script to run inference and save `.npy` files  Converts `.npy` to C arrays |
| `inference_strings.py` | use with the above file |

---

## 📹 YouTube Demo: https://youtu.be/8EbgY_wZ5Qs

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
- Tested with X-CUBE-AI and STM32H723ZG, STM32F401RE — portable to other targets


