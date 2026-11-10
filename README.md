# Topic 26 · Participant 1 data handoff

**Peer Review Score Prediction from Kazakh and English Abstracts**  
Programming for AI · AAI-2606M · Nulimu Arey

This folder contains the bilingual dataset, exploratory analysis, and verified paper-level splits for the group’s English/Kazakh comparison. Start with the [data quality report](results/data_quality_report.md) for the methods, figures, and results.

## Reproduce the outputs

Run from this directory using Python 3.12. The included raw CSVs allow the pipeline to run offline after dependency installation. If they are absent, the script downloads pinned source revisions and verifies their SHA-256 hashes.

```bash
python -m pip install -r requirements-data.txt
python prepare_data.py
```

Open the notebooks with the same Python environment in VS Code or Jupyter. Each notebook can run independently with this repository layout.

| Notebook | What it shows |
| --- | --- |
| [01 Data collection](notebooks/01_data_collection.ipynb) | Source provenance, label alignment, one-to-one join, and processed dataset |
| [02 EDA](notebooks/02_eda.ipynb) | Label balance, missingness, length distributions, duplicates, and varying scores for the same paper |
| [03 Data leakage](notebooks/03_data_leakage.ipynb) | Paper-level grouping, exact and near-duplicate checks, split balance, and comparison against random row splitting |

## Files for the modelling teammate

| File | Paired review rows | Use |
| --- | ---: | --- |
| `data/processed/merged_dataset.csv` | 415 | Full corpus, including 30 unlabeled rows |
| `data/splits/train.csv` | 270 | Fit the vectorizers and models |
| `data/splits/validation.csv` | 57 | Select settings and compare validation results |
| `data/splits/test.csv` | 58 | Run the final matched comparison |

Each row is one review, with both language versions stored as columns. Run the same split twice for monolingual experiments, switching the title and abstract columns. `review_id` aligns the predicted labels for the two languages.

| Column | Meaning |
| --- | --- |
| `review_id` | Published source row index shared by the two releases |
| `paper_id` | Stable identifier derived from the normalized English title, shared by every review of a paper |
| `label` | `Low`, `Moderate`, or `High`. Empty only in the merged corpus |
| `title_en`, `abstract_en` | English model features |
| `title_kk`, `abstract_kk` | Kazakh model features |
| `normalized_title_en`, `normalized_title_kk` | Normalized titles for auditing and grouping |
| `split_group_id`, `split` | Group and partition assignments present in the split files |

```python
import pandas as pd

def load_language(partition, language):
    rows = pd.read_csv(f"data/splits/{partition}.csv")
    text = rows[f"title_{language}"] + "\n\n" + rows[f"abstract_{language}"]
    return text, rows["label"], rows["review_id"]

X_en, y_en, ids_en = load_language("train", "en")
X_kk, y_kk, ids_kk = load_language("train", "kk")
assert ids_en.equals(ids_kk) and y_en.equals(y_kk)
```

Source releases are pinned in `src/data_collection.py`. [SIST English Corpus](https://huggingface.co/datasets/Arailym-tleubayeva/sist-english-corpus/viewer) and [SIST Kazakh Corpus](https://huggingface.co/datasets/Arailym-tleubayeva/sist-kazakh-corpus).

## Report and defense contribution

**Contribution statement**  
Nulimu Arey prepared and validated the parallel English–Kazakh review corpus, analyzed class balance, missingness and text lengths, and produced a reproducible train/validation/test partition with a paper-level leakage audit.

**Short defense script**  
“I aligned 415 bilingual review pairs by the published row ID and verified that their scores matched. I kept 385 scored pairs for supervised learning. Several papers have multiple reviews with different scores even though their title and abstract are identical. With a random row split, 37 papers appeared in both training and test. I assigned entire papers to a single partition and checked normalized titles, exact text and near-duplicate abstracts in both languages. The final splits contain 270, 57 and 58 scored pairs, with zero overlap under those checks. This gives both language models the same held-out review examples.”

**中文交接**  
把 `data/splits/` 中的三个 CSV 交给模型组员。英文模型用 `title_en`、`abstract_en`，哈语模型用 `title_kk`、`abstract_kk`，两边用同一 `label`。`review_id` 用来对齐两种语言的预测，`paper_id` 用来核对同一论文的多条评审。英文报告和图表可直接并入团队材料。
