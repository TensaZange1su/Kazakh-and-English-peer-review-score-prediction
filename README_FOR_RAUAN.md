# README — ML Model Integration Guide
## Peer Review Score Prediction from Kazakh and English Abstracts

**For:** Tasbolat Rauan  
**Project:** Peer Review Score Prediction from Kazakh and English Abstracts  
**Role:** Application Integration, Frontend and Database

---

## 1. Project Overview

The project aims to predict peer-review scores based on academic paper titles and abstracts written in Kazakh and English.

The ML component provides trained classification models that can be integrated into a Streamlit application.

The application should allow users to:

- Enter a paper title and abstract.
- Select the input language.
- Select a prediction model.
- Receive a predicted review score.
- View prediction history.
- Explore the original dataset.
- Compare model performance.
- Submit feedback on predictions.

The prediction classes are:

| Class | Description |
|---|---|
| Low | Low review score |
| Moderate | Moderate review score |
| High | High review score |

**Important:** The model predicts a reviewer's score based on the paper title and abstract. It does not determine the objective quality of a paper.

---

## 2. Available Machine Learning Models

There are 12 trained model configurations available.

### 2.1 TF-IDF Models

Location: `models/tfidf/`

| Model | File |
|---|---|
| English Word | `english_word.joblib` |
| English Character | `english_char.joblib` |
| English Combined | `english_combined.joblib` |
| Kazakh Word | `kazakh_word.joblib` |
| Kazakh Character | `kazakh_char.joblib` |
| Kazakh Combined | `kazakh_combined.joblib` |
| Multilingual Word | `multilingual_word.joblib` |
| Multilingual Character | `multilingual_char.joblib` |
| Multilingual Combined | `multilingual_combined.joblib` |

TF-IDF models contain the fitted vectorization and Logistic Regression pipeline.

They can be loaded using Joblib and called directly through `.predict()`.

### 2.2 Transformer Models

Location: `models/transformer/`

| Model | File |
|---|---|
| E5 English | `e5_english.joblib` |
| E5 Kazakh | `e5_kazakh.joblib` |
| E5 Multilingual | `e5_multilingual.joblib` |

These files contain the trained Logistic Regression classifiers.

The Transformer encoder is loaded separately:

`intfloat/multilingual-e5-small`

Hugging Face model page:

https://huggingface.co/intfloat/multilingual-e5-small

**Important:** The E5 encoder is not stored inside the `.joblib` classifier files. It must be loaded separately using Sentence Transformers.

---

## 3. Model Loading

### 3.1 TF-IDF

```python
import joblib
from pathlib import Path

MODEL_DIR = Path("models/tfidf")

tfidf_models = {
    "english_word": joblib.load(
        MODEL_DIR / "english_word.joblib"
    ),
    "english_char": joblib.load(
        MODEL_DIR / "english_char.joblib"
    ),
    "english_combined": joblib.load(
        MODEL_DIR / "english_combined.joblib"
    ),
    "kazakh_word": joblib.load(
        MODEL_DIR / "kazakh_word.joblib"
    ),
    "kazakh_char": joblib.load(
        MODEL_DIR / "kazakh_char.joblib"
    ),
    "kazakh_combined": joblib.load(
        MODEL_DIR / "kazakh_combined.joblib"
    ),
    "multilingual_word": joblib.load(
        MODEL_DIR / "multilingual_word.joblib"
    ),
    "multilingual_char": joblib.load(
        MODEL_DIR / "multilingual_char.joblib"
    ),
    "multilingual_combined": joblib.load(
        MODEL_DIR / "multilingual_combined.joblib"
    )
}
```

### 3.2 Transformer E5

```python
import joblib
import torch

from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

encoder = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device=DEVICE
)

transformer_models = {
    "e5_english": joblib.load(
        "models/transformer/e5_english.joblib"
    ),
    "e5_kazakh": joblib.load(
        "models/transformer/e5_kazakh.joblib"
    ),
    "e5_multilingual": joblib.load(
        "models/transformer/e5_multilingual.joblib"
    )
}
```

Load the encoder and classifiers once when the application starts. Do not reload them for every prediction.

For deployment without internet access, download the encoder in advance and configure loading from its local directory.

---

## 4. Prediction Logic

### 4.1 TF-IDF Prediction

TF-IDF models already contain their fitted preprocessing and classification pipeline.

Example:

```python
def predict_tfidf(text, model_name, models):
    model = models[model_name]

    prediction = model.predict([text])[0]

    return prediction
```

The input text should follow the same title and abstract preparation used during training.

### 4.2 Transformer E5 Prediction

During training, E5 inputs were formatted using the following structure:

```python
def prepare_e5_text(title, abstract):
    return (
        f"query: Title: {title} "
        f"Abstract: {abstract}"
    )
```

Generate an embedding:

```python
def predict_e5(
    title,
    abstract,
    classifier,
    encoder
):
    text = prepare_e5_text(title, abstract)

    embedding = encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    prediction = classifier.predict(embedding)[0]

    return prediction
```

For multilingual E5, use the same classifier with either an English or Kazakh input.

For language-specific E5 classifiers, select the corresponding classifier.

**Important:** Do not train or fit the models again during normal application usage. The application should use the existing trained weights.

---

## 5. Recommended Application Structure

```text
project/
│
├── app.py
│
├── pages/
│   ├── 1_Prediction.py
│   ├── 2_Prediction_History.py
│   ├── 3_Dataset_Explorer.py
│   ├── 4_Model_Performance.py
│   └── 5_Feedback.py
│
├── services/
│   ├── model_service.py
│   └── prediction_service.py
│
├── database/
│   ├── connection.py
│   ├── schema.sql
│   ├── init_db.py
│   ├── import_data.py
│   └── repository.py
│
├── models/
│   ├── tfidf/
│   └── transformer/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── results/
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

---

## 6. Required Frontend Functionality

### Page 1 — Prediction

The user should be able to:

1. Enter the paper title.
2. Enter the abstract.
3. Select English or Kazakh.
4. Select a model.
5. Click the Predict button.
6. View the predicted class.
7. View inference time.

Suggested model selection:

- TF-IDF English Word
- TF-IDF English Character
- TF-IDF English Combined
- TF-IDF Kazakh Word
- TF-IDF Kazakh Character
- TF-IDF Kazakh Combined
- TF-IDF Multilingual
- E5 English
- E5 Kazakh
- E5 Multilingual

The model list can be filtered by the selected language.

### Page 2 — Prediction History

Display previously submitted predictions.

Suggested columns:

| Field | Description |
|---|---|
| ID | Prediction ID |
| Title | Paper title |
| Language | Input language |
| Model | Selected model |
| Prediction | Predicted class |
| Inference Time | Prediction duration |
| Created At | Timestamp |

The user should be able to open individual records and inspect their input text.

### Page 3 — Dataset Explorer

Display the original dataset with:

- Paper title.
- Abstract.
- Review label.
- Language.
- Dataset split.

Add filters for labels, languages and splits.

### Page 4 — Model Performance

Display existing experimental results from the `results/` directory.

Recommended metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Inference Time
- Model Size

Use charts and comparison tables.

### Page 5 — Feedback

Allow users to submit a corrected label for a prediction.

Feedback should be stored separately from the original dataset.

Do not automatically retrain models using user feedback.

---

## 7. SQLite Database Requirements

All application predictions and user interactions should be persisted in SQLite.

Recommended tables:

| Table | Purpose |
|---|---|
| `papers` | Paper information |
| `reviews` | Original labeled reviews |
| `experiments` | Model configurations |
| `metrics` | Experiment evaluation results |
| `predictions` | Predictions made through the interface |
| `user_feedback` | User corrections |
| `activity_logs` | Application activity history |

Important distinction:

- `reviews` contains original ground-truth labels.
- `predictions` contains model-generated results.
- `user_feedback` contains user-submitted corrections.

Never overwrite original review labels with predicted labels.

Every new prediction should create a new database record.

---

## 8. Existing Data and Results

Use existing CSV and Parquet files instead of manually recreating the data.

Relevant sources include:

- `data/processed/merged_dataset.csv`
- `data/splits/train.csv`
- `data/splits/validation.csv`
- `data/splits/test.csv`
- Corresponding Parquet files.
- `results/tfidf_metrics.csv`
- `results/tfidf_detailed_metrics.csv`
- `results/tfidf_benchmark.csv`
- `results/transformer_metrics.csv`
- `results/transformer_benchmark.csv`
- Prediction and error-analysis exports.

Import these files into SQLite using Pandas.

Keep CSV and Parquet as source/export formats. SQLite should support application operations and SQL queries.

---

## 9. Technical Requirements

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Sentence Transformers
- PyTorch
- SQLite

Install dependencies:

```powershell
pip install -r requirements.txt
```

If the E5 model is not cached locally, the first startup requires internet access to download it.

Run the application:

```powershell
streamlit run app.py
```

---

## 10. Important Integration Notes

1. Load models once at application startup.
2. Use the existing trained `.joblib` files.
3. Do not retrain models during application usage.
4. Use the same E5 text formatting as in the experiments.
5. Apply embedding normalization for E5.
6. Keep original dataset labels separate from predictions.
7. Save every new prediction in SQLite.
8. Store inference time and timestamp.
9. Handle invalid input and model-loading errors.
10. Use relative paths based on the project root rather than hardcoded local Windows paths.

### Expected Result

The final application should provide an integrated system where users can submit English or Kazakh academic abstracts, receive predictions from trained TF-IDF or E5 models, explore model performance, review prediction history, and submit feedback.

All new predictions and user interactions should be stored persistently in SQLite.