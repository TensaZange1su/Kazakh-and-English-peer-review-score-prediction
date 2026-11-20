# Peer Review Score Prediction from Kazakh and English Abstracts

A bilingual machine learning project for predicting peer-review score categories from scientific paper titles and abstracts in Kazakh and English.

The project compares lightweight TF-IDF classifiers with multilingual Transformer-based embeddings and integrates the trained models into a Streamlit application for interactive prediction, model comparison, analytics, and prediction history.

---

## Project Overview

Peer review is an important part of scientific publishing. Reviewers evaluate submitted papers based on their relevance, methodology, originality, clarity, and other criteria. These assessments are often represented by categorical scores.

This project investigates whether patterns in scientific paper titles and abstracts can be used to predict recorded peer-review score categories.

The task is formulated as a **multiclass text classification problem** with three target classes:

- **Low**
- **Moderate**
- **High**

The project focuses on Kazakh and English scientific texts and compares traditional statistical text representations with contextual multilingual embeddings.

The model predicts a recorded review category. It does not directly measure the scientific quality of a paper and should not replace human peer review.

---

## Objectives

The main objectives of the project are:

1. Construct and validate a bilingual Kazakh-English review dataset.
2. Perform exploratory data analysis and investigate data quality.
3. Develop text classification models using Word-level, Character-level, and Combined TF-IDF features.
4. Evaluate multilingual Transformer embeddings using the E5 model.
5. Compare language-specific and multilingual model configurations.
6. Evaluate models using class-sensitive metrics.
7. Analyze classification errors and computational performance.
8. Integrate trained models into a web application.
9. Provide an interactive interface for prediction, model comparison, analytics, and prediction history.

---

## Dataset

### Data Sources

The project uses the following datasets:

- SIST English Corpus
- SIST Kazakh Corpus

The source datasets were aligned to create a bilingual corpus containing corresponding review records.

### Dataset Statistics

| Statistic | Value |
|---|---:|
| Bilingual review pairs | 415 |
| Unique papers | 215 |
| Labeled review pairs | 385 |
| Labeled papers | 185 |
| Unlabeled review pairs | 30 |

The 30 unlabeled records are retained in the merged corpus but excluded from supervised training and evaluation.

### Target Classes

The original review-score categories are mapped into three classification labels:

| Original category | Target label |
|---|---|
| Төмен | Low |
| Орташа | Moderate |
| Жоғары | High |

### Class Distribution

| Class | Samples | Percentage |
|---|---:|---:|
| Low | 44 | 11.4% |
| Moderate | 238 | 61.8% |
| High | 103 | 26.8% |
| **Total** | **385** | **100%** |

The dataset is imbalanced, with Moderate being the dominant category.

This imbalance is taken into account during model evaluation. Accuracy alone is not considered sufficient, so Macro F1 is used as an important comparison metric.

---

## Data Preprocessing

The preprocessing workflow includes:

1. Loading the English and Kazakh source datasets.
2. Aligning bilingual review records.
3. Validating paired records and score mappings.
4. Checking missing values and text quality.
5. Investigating repeated papers and multiple reviewer scores.
6. Preparing model input from paper titles and abstracts.
7. Excluding unlabeled records from supervised experiments.
8. Creating paper-level train, validation, and test splits.

### Model Input

The models use:

- Paper title
- Paper abstract

The following information is excluded from model features:

- Review comments
- Keywords
- Citation counts
- Source identifiers

The prediction is therefore based only on the textual content of the title and abstract.

### Repeated Reviews

Multiple review records may correspond to the same scientific paper.

In 93 of the 185 labeled papers, more than one score category was recorded for the same title and abstract.

These observations were retained because the target represents an individual reviewer's recorded category rather than a single consensus score.

This introduces an inherent ambiguity: identical textual inputs may have different target labels.

---

## Train / Validation / Test Split

To reduce data leakage, records were divided at the paper level rather than randomly at the row level.

| Partition | Review pairs | Papers |
|---|---:|---:|
| Train | 270 | 130 |
| Validation | 57 | 27 |
| Test | 58 | 28 |
| **Total** | **385** | **185** |

### Leakage Prevention

The dataset contains multiple records associated with the same paper.

A random row-level split could place records from the same paper in both training and testing partitions. This would allow the model to encounter similar or identical text during training and evaluation.

The project therefore uses grouped splitting based on paper identity.

Additional checks were performed using:

- Normalized titles
- Exact titles
- Exact abstracts
- Near-duplicate text detection

Near-duplicate detection was based on character 5-gram Jaccard similarity with a threshold of 0.80.

The resulting split was designed to prevent paper-level overlap between training, validation, and test sets.

---

## Methodology

The project evaluates two main approaches to text representation.

### 1. TF-IDF

TF-IDF converts text into sparse numerical feature vectors based on term frequency and inverse document frequency.

The experiments include:

- Word-level TF-IDF
- Character-level TF-IDF
- Combined Word and Character TF-IDF

Each representation is followed by a Logistic Regression classifier.

### Word-level TF-IDF

Word-level features represent individual words and word sequences.

The experiments use word n-grams, allowing the model to capture:

- Individual terms
- Short phrases
- Frequent lexical patterns

Word-level representations are useful for identifying specific terminology in scientific abstracts.

### Character-level TF-IDF

Character-level features represent sequences of characters rather than complete words.

Character n-grams can capture:

- Word fragments
- Morphological patterns
- Suffixes
- Orthographic similarities
- Variations in word forms

This approach is particularly relevant for Kazakh, which has rich morphology and many word forms generated through suffixation.

### Combined TF-IDF

Combined TF-IDF merges word-level and character-level features.

The objective is to combine lexical information with subword and morphological patterns.

### Classifier: Logistic Regression

Logistic Regression is used as the classification algorithm for TF-IDF representations.

It is suitable for:

- High-dimensional sparse features
- Small datasets
- Multiclass classification
- Efficient training and inference

The classifier predicts one of the three target categories: Low, Moderate, or High.

---

## 2. Multilingual E5

The second approach uses a pretrained multilingual Transformer encoder:

**Model:** `intfloat/multilingual-e5-small`

The encoder converts text into dense contextual embeddings.

Unlike TF-IDF, which represents text through statistical lexical features, E5 embeddings capture contextual semantic information.

### Embedding Pipeline

The input text is formatted as:

```text
query: Title: {title} Abstract: {abstract}
```

The E5 encoder produces normalized dense embeddings.

These embeddings are then used as input to a separate Logistic Regression classifier.

### Important

The project does not fine-tune the entire Transformer encoder.

Instead:

1. The pretrained E5 model generates embeddings.
2. The embeddings are extracted from the scientific text.
3. Logistic Regression is trained on the embeddings.
4. The classifier predicts Low, Moderate, or High.

### E5 Configurations

Three configurations were evaluated:

- E5 English
- E5 Kazakh
- E5 Multilingual

The classifier regularization parameter `C` was selected from:

```text
0.1
1.0
10.0
```

Configurations with and without balanced class weights were evaluated using validation Macro F1.

The selected classifier was refitted using the training and validation embeddings.

---

## Experiments

The experiments compare language-specific and multilingual configurations.

### TF-IDF Experiments

| Language | Representation |
|---|---|
| English | Word |
| English | Character |
| English | Combined |
| Kazakh | Word |
| Kazakh | Character |
| Kazakh | Combined |
| Multilingual | Word |
| Multilingual | Character |
| Multilingual | Combined |

### E5 Experiments

| Configuration | Encoder |
|---|---|
| English E5 | Multilingual E5-small |
| Kazakh E5 | Multilingual E5-small |
| Multilingual E5 | Multilingual E5-small |

The language-specific and multilingual experiments use the same pretrained encoder but different text configurations and classifier evaluation setups.

---

## Evaluation Metrics

The models are evaluated using:

### Accuracy

The proportion of correct predictions among all predictions.

\[
Accuracy=\frac{Correct\ Predictions}{Total\ Predictions}
\]

### Precision

Measures how many predictions for a class are correct.

### Recall

Measures how many actual examples of a class are successfully identified.

### Macro Precision

The average Precision across all classes, treating each class equally.

### Macro Recall

The average Recall across all classes, treating each class equally.

### Macro F1

The average F1-score across Low, Moderate, and High.

Macro F1 is particularly important because the dataset is imbalanced.

### Weighted F1

The average F1-score weighted by the number of samples in each class.

---

## Experimental Results

### TF-IDF Results

| Configuration | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| English Word | 0.5172 | 0.2350 | 0.2970 | 0.2552 | 0.4348 |
| English Char | 0.4483 | 0.3389 | 0.3310 | 0.3316 | 0.4500 |
| English Combined | 0.6034 | 0.2011 | 0.3333 | 0.2509 | 0.4542 |
| Kazakh Word | 0.5862 | 0.2024 | 0.3238 | 0.2491 | 0.4509 |
| **Kazakh Char** | **0.5345** | **0.4917** | **0.4012** | **0.4124** | **0.5165** |
| Kazakh Combined | 0.5862 | 0.2024 | 0.3238 | 0.2491 | 0.4509 |
| Multilingual Word | 0.5517 | 0.2314 | 0.3104 | 0.2545 | 0.4452 |
| Multilingual Char | 0.4828 | 0.3418 | 0.3423 | 0.3408 | 0.4710 |
| Multilingual Combined | 0.5172 | 0.2611 | 0.3083 | 0.2762 | 0.4496 |

**Best TF-IDF configuration: Kazakh Character TF-IDF**

- Accuracy: 0.5345
- Macro F1: 0.4124
- Weighted F1: 0.5165

### E5 Results

| Model | Validation Macro F1 | Test Accuracy | Test Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| E5 English | 0.4720 | 0.3793 | 0.3169 | 0.4125 |
| E5 Kazakh | 0.3446 | 0.3966 | 0.2983 | 0.4088 |
| **E5 Multilingual** | **0.4708** | **0.4914** | **0.3396** | **0.4839** |

**Best E5 configuration: E5 Multilingual**

- Accuracy: 0.4914
- Macro F1: 0.3396
- Weighted F1: 0.4839

### Overall Comparison

The strongest reported test Macro F1 was achieved by Kazakh Character TF-IDF.

| Model | Test Macro F1 |
|---|---:|
| **Kazakh Character TF-IDF** | **0.4124** |
| Multilingual Character TF-IDF | 0.3408 |
| E5 Multilingual | 0.3396 |
| E5 English | 0.3169 |
| E5 Kazakh | 0.2983 |

The results demonstrate that a lightweight character-based representation can outperform a larger pretrained Transformer model in a small-data setting.

---

## Error Analysis

Error analysis was conducted to investigate common model mistakes.

For the E5 English configuration:

- Test samples: 58
- Correct predictions: 22
- Incorrect predictions: 36
- Accuracy: 0.3793

### Frequent Confusion Patterns

| Actual | Predicted | Errors |
|---|---|---:|
| Moderate | High | 11 |
| Moderate | Low | 9 |
| High | Low | 6 |
| Low | Moderate | 6 |
| High | Moderate | 4 |

The models frequently confuse neighboring review categories.

Potential reasons include:

- Small dataset size
- Class imbalance
- Subjective reviewer judgments
- Different reviewers assigning different categories to identical text
- Limited information available in titles and abstracts
- Absence of full manuscripts and reviewer-specific context

The model therefore learns statistical patterns associated with recorded review categories rather than objective scientific quality.

---

## Computational Benchmarking

The project also evaluates inference performance.

### TF-IDF Benchmark

Approximate time per 1,000 texts:

| Configuration | Time |
|---|---:|
| Kazakh Word TF-IDF | 85.4 ms |
| Kazakh Character TF-IDF | 763.3 ms |
| Multilingual Combined TF-IDF | 866.8 ms |

### E5 CUDA Benchmark

Approximate time per 1,000 texts:

| Configuration | Time |
|---|---:|
| E5 English | 1,159.9 ms |
| E5 Kazakh | 1,270.0 ms |
| E5 Multilingual | 1,244.0 ms |

### Application CPU Benchmark

A separate application benchmark was conducted on CPU using three repeats of 1,000 texts.

| Model family | Approximate time per 1,000 texts |
|---|---:|
| TF-IDF | 0.076 - 0.804 seconds |
| E5 | 9.64 - 10.63 seconds |

These benchmarks were performed under different conditions and should not be directly compared with the CUDA benchmark.

---

## System Architecture

The repository consists of three main workstreams.

### 1. Data Preparation

- Dataset collection
- Bilingual alignment
- Data validation
- Exploratory data analysis
- Class distribution analysis
- Repeated-review analysis
- Data leakage checks
- Grouped data splitting

### 2. Modeling and Evaluation

- TF-IDF feature extraction
- Word-level models
- Character-level models
- Combined feature models
- Logistic Regression
- Multilingual E5 embeddings
- Validation and test evaluation
- Error analysis
- Benchmarking
- Model export

### 3. Application Integration

- Streamlit interface
- Model loading and prediction
- Language and model selection
- SQLite persistence
- Prediction history
- Model comparison
- Analytics
- Automated integration testing

---

## Web Application

The project includes a Streamlit application that integrates the trained models.

### Application Pages

| Page | Description |
|---|---|
| Home | Project overview and navigation |
| Prediction | Predict peer-review category from title and abstract |
| Model Comparison | Compare model performance and benchmarks |
| Analytics | Explore experiment results and statistics |
| Prediction History | View previously saved predictions |

### Prediction Workflow

1. Enter a scientific paper title.
2. Enter the abstract.
3. Select the language.
4. Select a model.
5. Run prediction.
6. View the predicted category.
7. The successful prediction is saved to SQLite.

The application loads saved models and does not retrain them during normal usage.

### Database

SQLite is used for persistence.

The database contains two principal tables:

- `experiments`
- `predictions`

Experiment results are imported from CSV files using an idempotent import process.

Prediction records include information such as:

- Input title
- Abstract
- Language
- Model identifier
- Predicted category
- Inference duration
- Timestamp

Confidence values are withheld because probability calibration has not been validated.

---

## Tech Stack

### Programming

- Python

### Machine Learning

- scikit-learn
- Logistic Regression
- TF-IDF
- Sentence Transformers
- Multilingual E5
- PyTorch

### Data Processing

- pandas
- NumPy

### Experiment Environment

- Jupyter Notebook

### Application

- Streamlit

### Database

- SQLite

### Hardware

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 7700X |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| GPU Memory | 16 GB VRAM |
| RAM | 32 GB DDR5 |

CUDA-enabled GPU execution was used for supported neural-model workloads.

---

## Repository Structure

The repository contains datasets, notebooks, trained model artifacts, experiment results, and the integrated application.

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 04_tfidf_baseline.ipynb
│   └── 05_transformer_experiments.ipynb
│
├── models/
│   ├── tfidf/
│   └── transformer/
│
├── results/
│   ├── data_quality_report.md
│   ├── tfidf_detailed_metrics.csv
│   ├── transformer_metrics.csv
│   ├── tfidf_benchmark.csv
│   ├── transformer_benchmark.csv
│   ├── transformer_model_size.csv
│   └── error_analysis/
│
├── rauan/
│   ├── app.py
│   ├── APPLICATION.md
│   ├── requirements.txt
│   ├── services/
│   ├── repository/
│   ├── tests/
│   ├── data/
│   └── results/
│
├── README.md
└── ...
```

*Note: The structure above represents the main project components. Additional scripts, artifacts, and supporting files are included in the repository.*

---

## Installation

### Requirements

- Python
- pip
- Git

For Transformer inference, the pretrained E5 encoder must be available locally or downloaded from Hugging Face.

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Create a Virtual Environment

#### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Application Dependencies

```bash
pip install -r rauan/requirements.txt
```

---

## Running the Application

From the repository root:

```bash
cd rauan
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## Transformer Model Setup

The E5 model used by the project:

```text
intfloat/multilingual-e5-small
```

The model can be downloaded and cached automatically on the first Transformer prediction, provided internet access is available.

For an offline setup, the encoder can be downloaded in advance:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-small")
model.save("models/transformer/encoder")
```

The application can use a local encoder path through the `E5_MODEL_PATH` environment variable.

GPU execution can be configured using:

```bash
E5_DEVICE=cuda
```

CPU inference is used by default unless CUDA is explicitly configured.

---

## Testing

The repository includes automated integration tests covering:

- TF-IDF prediction parity
- E5 classifier-head checks
- Input validation
- Training input formatting
- SQLite persistence
- Experiment import
- Streamlit prediction-to-history workflow

The documented integration test suite reports **7 passing tests** in the tested environment.

Run the application tests according to the instructions in:

```text
rauan/APPLICATION.md
```

The E5 adapter tests use reference embeddings. A full fresh-encoder benchmark is a separate verification step.

---

## Team Contributions

### Nulimu Arey - Data Preparation

- Collected and aligned English and Kazakh review records.
- Validated paired records and label mapping.
- Conducted exploratory data analysis.
- Investigated class distribution and missingness.
- Analyzed text lengths and repeated reviews.
- Investigated potential data leakage.
- Created paper-level train, validation, and test splits.
- Documented the data-quality audit.

### Kuratov Almas - Modeling and Evaluation

- Developed Word-level, Character-level, and Combined TF-IDF classifiers.
- Implemented Logistic Regression classification pipelines.
- Conducted English, Kazakh, and multilingual experiments.
- Evaluated multilingual E5 embeddings with Logistic Regression classifier heads.
- Compared models using Accuracy, Macro Precision, Macro Recall, Macro F1, and Weighted F1.
- Conducted error analysis and investigated confusion patterns.
- Evaluated inference performance and model size.
- Exported trained model artifacts and evaluation results for application integration.

### Tasbolat Rauan - Application Integration

- Integrated trained TF-IDF pipelines and E5 classifier heads into a Streamlit application.
- Implemented English and Kazakh prediction workflows.
- Developed model selection and prediction functionality.
- Implemented SQLite persistence.
- Developed prediction history.
- Implemented model comparison and analytics views.
- Added input validation and automated integration tests.
- Documented setup and demonstration instructions.

---

## Limitations

The project has several important limitations:

1. **Small dataset**  
   The labeled corpus contains only 385 review pairs.

2. **Class imbalance**  
   The Low category represents only 11.4% of labeled observations.

3. **Subjective labels**  
   Different reviewers may assign different categories to the same paper.

4. **Limited input information**  
   Models use only titles and abstracts.

5. **No reviewer-specific context**  
   Reviewer preferences and individual evaluation criteria are not modeled.

6. **No calibrated confidence**  
   Model probabilities have not been validated for calibration.

7. **Limited generalization**  
   Performance on papers from other conferences, disciplines, or datasets remains unverified.

8. **Application security**  
   The research application does not include authentication or user-specific data separation and should not be exposed publicly with private manuscript content.

---

## Conclusion

This project developed and evaluated bilingual text classification models for predicting peer-review score categories from scientific paper titles and abstracts.

The experiments compared traditional TF-IDF representations with multilingual E5 embeddings.

The best overall reported result was achieved by **Kazakh Character TF-IDF**, with a test Macro F1 of **0.4124**.

The multilingual E5 model achieved a test Macro F1 of **0.3396**.

The results suggest that character-level features can be highly effective for small datasets and morphologically rich languages. A more complex Transformer-based representation does not necessarily outperform a lightweight statistical baseline when labeled training data are limited.

The project also demonstrates an end-to-end workflow from bilingual dataset preparation and leakage-aware evaluation to model integration in a web application.

The system is intended as an experimental decision-support tool and does not replace expert peer review.

---

## Authors

**Kuratov Almas**  
Modeling and Evaluation

**Tasbolat Rauan**  
Application Integration

**Nulimu Arey**  
Data Preparation

**Group:** AAI-2606M

**Course:** Programming for AI

**Topic / Cluster:** Lightweight Transformer Models for Kazakh Text
