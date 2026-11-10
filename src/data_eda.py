"""Summarize corpus quality and plot per-paper text statistics."""

import pandas as pd

from src.data_collection import LABELS, ROOT, SOURCES, token_count

RESULTS = ROOT / "results"
FIGURE = RESULTS / "corpus_overview.png"


def raw_missingness(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({lang: frame.isna().sum() for lang, frame in frames.items()})


def label_counts(reviews: pd.DataFrame) -> pd.Series:
    return reviews["label"].fillna("Unlabeled").value_counts().reindex([*LABELS, "Unlabeled"], fill_value=0)


def paper_lengths(reviews: pd.DataFrame) -> pd.DataFrame:
    # Retain one text per paper for EDA. Count reviews separately for class balance.
    fields = [f"{field}_{lang}" for field in ("title", "abstract") for lang in SOURCES]
    paper_text = reviews[["paper_id", *fields]].drop_duplicates()
    if not paper_text["paper_id"].is_unique:
        raise ValueError("A paper has multiple distinct title/abstract versions")
    return paper_text.set_index("paper_id").apply(lambda column: column.map(token_count))


def length_summary(lengths: pd.DataFrame) -> pd.DataFrame:
    summary = lengths.describe(percentiles=[0.25, 0.50, 0.75]).T
    iqr = summary["75%"] - summary["25%"]
    outliers = lengths.lt(summary["25%"] - 1.5 * iqr) | lengths.gt(summary["75%"] + 1.5 * iqr)
    summary["IQR outliers"] = outliers.sum()
    return summary


def review_patterns(reviews: pd.DataFrame) -> dict[str, int]:
    labeled = reviews.dropna(subset=["label"])
    return {
        "review pairs": len(reviews),
        "papers": reviews["paper_id"].nunique(),
        "labeled review pairs": len(labeled),
        "labeled papers": labeled["paper_id"].nunique(),
        "papers with repeated labeled reviews": int(labeled["paper_id"].value_counts().gt(1).sum()),
        "papers with differing labeled scores": int(labeled.groupby("paper_id")["label"].nunique().gt(1).sum()),
    }


def plot_overview(reviews: pd.DataFrame, lengths: pd.DataFrame, missing: pd.DataFrame) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 7), layout="constrained", facecolor="#f5f8fc")
    FigureCanvasAgg(figure)
    figure.suptitle("SIST bilingual corpus | Data quality overview", fontsize=18, fontweight="bold", color="#152743")
    axes = figure.subplots(2, 2)
    counts = label_counts(reviews)
    axis = axes[0, 0]
    bars = axis.bar(counts.index, counts.values, color=["#6a5b9d", "#176f81", "#3476b7", "#9da9ba"])
    axis.bar_label(bars, padding=3)
    axis.set(ylim=(0, counts.max() * 1.20), ylabel="Paired review records")

    axis = axes[0, 1]
    columns = ["Keywords", "CitationCount", "Review_score", "Review_comment"]
    bars = axis.barh(["Keywords", "Citation count", "Review score", "Review comment"], missing.loc[columns, "en"], color="#3476b7")
    axis.bar_label(bars, padding=4)
    axis.invert_yaxis()
    axis.set(xlim=(0, 270), xlabel="Missing records per language")

    for index, field in enumerate(("title", "abstract")):
        axis = axes[1, index]
        data = [lengths[f"{field}_{lang}"] for lang in SOURCES]
        boxes = axis.boxplot(data, tick_labels=["English", "Kazakh"], orientation="horizontal", patch_artist=True,
                             medianprops={"color": "#ffffff", "linewidth": 2})
        for box, color in zip(boxes["boxes"], ("#3476b7", "#176f81")):
            box.set_facecolor(color)
        axis.invert_yaxis()
        axis.set(xlabel="Unicode word tokens per paper")

    titles = ["A. Published review scores", "B. Source fields with missing values",
              "C. Title lengths", "D. Abstract lengths"]
    for axis, title in zip(axes.flat, titles):
        axis.set_axisbelow(True)
        axis.grid(axis="y" if axis is axes[0, 0] else "x", color="#e5edf5")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(title, loc="left", fontweight="bold", color="#152743")
    RESULTS.mkdir(exist_ok=True)
    figure.savefig(FIGURE, dpi=150)
