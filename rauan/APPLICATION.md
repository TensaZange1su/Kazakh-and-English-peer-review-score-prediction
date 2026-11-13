# Participant 3: integrated application

The Streamlit app uses all nine saved TF-IDF pipelines and three E5 classifier heads. It never refits models. The five views are Home, Prediction, Model Comparison, Analytics, and Prediction History.

## Setup and run

Python 3.12 is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r rauan/requirements.txt
cd rauan
streamlit run app.py
```

The application creates `data/application.sqlite3` and idempotently imports existing experiment CSVs. Set `APP_DB_PATH` to choose another persistent location. Run Streamlit, tests, and scripts from the `rauan/` directory. The database and local benchmarks live under `rauan/data/` and `rauan/results/`; shared datasets, trained classifiers, and original experiment results stay at the repository root. All data and artifact paths are resolved from the source directory.

For E5, Sentence Transformers loads `intfloat/multilingual-e5-small`, including its tokenizer, on the first Transformer prediction. This requires a cached model or internet access and substantial disk/memory. To prepare an offline demo, download the encoder in advance:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small').save('models/transformer/encoder')"
export E5_MODEL_PATH="$PWD/models/transformer/encoder"
```

The service automatically prefers `models/transformer/encoder` when present. A verified local copy was prepared in this workspace; it is ignored by Git. `E5_MODEL_PATH` overrides that default.

`E5_DEVICE` defaults to `cpu`; set it to `cuda` for an available NVIDIA GPU. The training notebook used a 512-token limit; long E5 abstracts are truncated accordingly. TF-IDF does not need the encoder or network access. Load only trusted joblib artifacts.

### Artifact compatibility

The supplied joblib files record scikit-learn **1.9.1**, unavailable on the package index used during integration. The requirements select **1.6.1** for compatibility testing. Scikit-learn does not guarantee loading across versions. Do not suppress version warnings. Run the parity tests below in the deployment environment before a demonstration, and obtain the original environment or re-exported artifacts from Participant 2 if parity fails. No artifacts are modified by the application.

## Unified interface

```python
from services.model_service import predict, ModelService
result = predict('An abstract in English.', 'en')
result = ModelService('e5_kazakh').predict('Қазақша аннотация.', 'kk', title='Тақырып')
```

The default is the language-specific character TF-IDF pipeline. `available_models('en')` and `available_models('kk')` enumerate compatible configurations. The service returns the predicted class, model ID, input, and warm inference duration. The UI persists successful predictions with UTC timestamps. Empty abstracts, non-string inputs, unknown languages, incompatible models, and inputs over 100,000 characters are rejected. An empty title is supported.

Preprocessing exactly matches the modeling notebooks: TF-IDF uses `title + ' [SEP] ' + abstract`; E5 uses `query: Title: {title} Abstract: {abstract}` with normalized embeddings. Input content is not lowercased or stripped outside the fitted pipelines. The model outputs categories, not numeric review scores. Confidence is withheld because the supplied logistic-regression probabilities have no documented calibration validation.

## Storage

`database/schema.sql` defines indexed predictions and experiment records. Experiment imports are keyed by model, evaluation language, and source; reloading updates results without duplicating them. Predictions are append-only. Source reviews, trained artifacts, and original labels are never overwritten. SQL uses parameterized values; each operation opens and closes its own transactional connection. History shows the most recent 500 entries, filters by language, exposes full inputs, and supports CSV export.

The comparison dashboard reads stored experiment results. Analytics aligns the `test_*` Transformer metrics with TF-IDF metrics and explicitly separates combined and single-language evaluation. Existing confusion matrices are available. The imported E5 measurements were obtained on CUDA; they are historical results, not measurements on the current machine. Source “MB” values use 1024² bytes (MiB). Encoder repository snapshot size and loaded weight memory are shown separately.

## Verification and benchmarking

Run these commands from `rauan/` with the virtual environment activated.

```bash
python -m unittest discover -s tests -v
python -m scripts.benchmark --models english_char kazakh_char multilingual_char
# Verify freshly generated E5 embeddings and labels against the handoff:
python -m scripts.verify_encoder
# All 12 configurations, including real E5 encoder inference:
python -m scripts.benchmark
```

The benchmark repeats held-out test examples to exactly 1,000 texts per model; they are not 1,000 independent papers. Multilingual workloads include both languages. It warms the model, runs three repeats, and reports median batch duration excluding loading, with E5 tokenization/encoding/classification included. CUDA synchronization makes GPU timing synchronous. It records device, host platform, classifier artifact size, encoder parameter count, and weight memory. Encoder directory storage counts the local encoder directory or the active cached snapshot, deduplicating linked files. Each completed model is written to `results/application_benchmark.csv`, then imported into SQLite after the command completes. Refresh/restart the app after benchmarking to import new results.

Integration tests compare every language-specific TF-IDF held-out prediction against Participant 2’s export and test SQLite persistence, Unicode inputs, validation, and the Streamlit prediction-to-history workflow. E5 adapter tests use real saved classifiers and reference embeddings; they do **not** verify freshly generated encoder embeddings. Run the full benchmark with the downloaded encoder to exercise real E5 inference.

## Demo and deployment

1. Install dependencies and run verification on the presentation machine.
2. Pre-download E5 and run the benchmark before the presentation if demonstrating Transformers.
3. Start Streamlit, explain the project on Home, then submit English and Kazakh examples on Prediction.
4. Open Prediction History and inspect the saved title and abstract.
5. Show Model Comparison, distinguishing local timings from the imported CUDA timings, then explore Analytics.

Deploy with a Python runtime, `requirements.txt`, repository artifacts, and a persistent writable volume for `APP_DB_PATH`. The default Streamlit port is 8501. An ephemeral hosting filesystem loses SQLite records on restart; use a persistent disk for retained history. This research demo has no authentication or user separation, so every app visitor can see saved prediction inputs. Do not expose private submissions publicly. No live deployment is included.

## Integration validation on this machine

The seven automated tests passed on Python 3.9 with scikit-learn 1.6.1. All nine TF-IDF models matched the handoff predictions for each supported language on the held-out set; all three E5 classifier heads matched the exported test accuracy and macro F1 using reference embeddings. The Streamlit test submitted a prediction, found it in history, and opened the remaining pages. Dependency resolution passed `pip check`.

Local CPU benchmarks for all nine TF-IDF models are included in `results/application_benchmark.csv` (three repeats of 1,000 texts each). They ranged from about 0.076 to 0.804 seconds per batch. These timings exclude artifact loading. The first UI test encountered slow initial dependency imports; the documented test now allows up to 180 seconds for cold startup.

This environment emits scikit-learn artifact-version warnings and NumPy matrix-operation runtime warnings during E5 reference-head evaluation, although the exported metrics match. Python 3.12 deployment has not been tested here. Validate the original modeling environment before relying on results beyond this research demonstration.

The real E5 encoder was also downloaded and checked offline: all 58 English and 58 Kazakh reference embeddings matched within numerical tolerance, and both language-specific and multilingual head predictions matched exactly.

All three real E5 CPU benchmarks also completed with three repeats of 1,000 texts: English 9.964 s, Kazakh 10.626 s, multilingual 9.644 s. The encoder has 117,653,760 parameters and 448.813 MiB of loaded parameter weights; the active cached snapshot measured 470.441 MiB. All 12 local benchmark rows were imported into SQLite. The offline encoder copy occupies approximately 466 MiB and includes the tokenizer.
