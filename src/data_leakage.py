"""Group bilingual records by paper and audit exact and near-duplicate overlap."""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from src.data_collection import LABELS, SOURCES, normalize

SPLITS = ("train", "validation", "test")
RATIOS = np.array([0.70, 0.15, 0.15])
SEED = 26
N_CANDIDATES = 2000
NEAR_THRESHOLD = 0.80
GRAM_SIZE = 5


def shingles(text: str) -> set[str]:
    return {text[i:i + GRAM_SIZE] for i in range(len(text) - GRAM_SIZE + 1)}


def find_duplicate_edges(reviews: pd.DataFrame) -> pd.DataFrame:
    """Find cross-paper matches in either language before any split is assigned."""
    edges = set()
    for language in SOURCES:
        for field in ("title", "abstract"):
            column = f"{field}_{language}"
            documents = reviews[["paper_id", column]].copy()
            documents[column] = documents[column].map(normalize)
            documents = documents.drop_duplicates()
            for _, matching in documents.groupby(column, sort=False):
                for left, right in combinations(sorted(matching["paper_id"]), 2):
                    edges.add((left, right, language, f"exact_{field}", 1.0))

        abstracts = reviews[["paper_id", f"abstract_{language}"]].copy()
        abstracts[f"abstract_{language}"] = abstracts[f"abstract_{language}"].map(normalize)
        abstracts = abstracts.drop_duplicates().sort_values("paper_id")
        texts = [(paper, text, shingles(text)) for paper, text in abstracts.itertuples(index=False, name=None)]
        for (left, left_text, left_grams), (right, right_text, right_grams) in combinations(texts, 2):
            if left == right or left_text == right_text or not left_grams or not right_grams:
                continue
            if min(len(left_grams), len(right_grams)) < NEAR_THRESHOLD * max(len(left_grams), len(right_grams)):
                continue
            shared = len(left_grams & right_grams)
            similarity = shared / (len(left_grams) + len(right_grams) - shared)
            if NEAR_THRESHOLD <= similarity < 1:
                edges.add((left, right, language, "near_abstract", round(similarity, 6)))
            elif similarity == 1:
                edges.add((left, right, language, "same_abstract_shingles", 1.0))
    return pd.DataFrame(sorted(edges), columns=["paper_a", "paper_b", "language", "check", "similarity"])


def duplicate_groups(paper_ids: pd.Series, edges: pd.DataFrame) -> dict[str, str]:
    """Return the connected components of the paper similarity graph."""
    neighbors = defaultdict(set)
    for left, right in edges[["paper_a", "paper_b"]].itertuples(index=False, name=None):
        neighbors[left].add(right)
        neighbors[right].add(left)
    assignments = {}
    for paper in sorted(paper_ids.unique()):
        if paper in assignments:
            continue
        group_id = "g_" + paper.removeprefix("p_")
        frontier = [paper]
        assignments[paper] = group_id
        while frontier:
            current = frontier.pop()
            for neighbor in neighbors[current]:
                if neighbor not in assignments:
                    assignments[neighbor] = group_id
                    frontier.append(neighbor)
    return assignments


def group_split(reviews: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Choose a reproducible group shuffle close to the target class proportions."""
    data = reviews.dropna(subset=["label"]).copy()
    data["split_group_id"] = data["paper_id"].map(groups)
    counts = pd.crosstab(data["split_group_id"], data["label"]).reindex(columns=LABELS, fill_value=0)
    values = counts.to_numpy()
    totals = values.sum(axis=0)
    targets = RATIOS[:, None] * totals
    boundaries = np.rint(np.cumsum(RATIOS)[:-1] * len(counts)).astype(int)
    rng = np.random.default_rng(SEED)
    best_loss = np.inf
    best = None
    for _ in range(N_CANDIDATES):
        candidate = np.split(rng.permutation(len(counts)), boundaries)
        observed = np.stack([values[indices].sum(axis=0) for indices in candidate])
        if (observed == 0).any():
            continue
        loss = np.square((observed - targets) / targets).sum()
        if loss < best_loss:
            best_loss, best = loss, candidate
    if best is None:
        raise ValueError("No group split contains all labels in each partition")
    group_to_split = {
        group: split for split, indices in zip(SPLITS, best) for group in counts.index[indices]
    }
    data["split"] = data["split_group_id"].map(group_to_split)
    return data


def random_row_split(data: pd.DataFrame) -> pd.Series:
    """Create a deliberately naive comparison split with the same row counts."""
    permutation = np.random.default_rng(SEED).permutation(data.index)
    sizes = data["split"].value_counts().reindex(SPLITS)
    chunks = np.split(permutation, sizes.cumsum().to_numpy()[:-1])
    return pd.Series({index: split for split, rows in zip(SPLITS, chunks) for index in rows})


def audit_overlap(data: pd.DataFrame, split_labels: pd.Series, edges: pd.DataFrame) -> pd.DataFrame:
    """Count intersections between each split pair using independent text keys."""
    metrics = {"paper_id": data["paper_id"], "split_group_id": data["split_group_id"]}
    for language in SOURCES:
        for field in ("title", "abstract", "normalized_title"):
            column = f"{field}_{language}"
            metrics[column] = data[column].map(normalize)
    rows = []
    for left, right in combinations(SPLITS, 2):
        left_mask = split_labels.reindex(data.index).eq(left)
        right_mask = split_labels.reindex(data.index).eq(right)
        row = {"split_a": left, "split_b": right}
        for key, series in metrics.items():
            row[key] = len(set(series[left_mask]) & set(series[right_mask]))
        papers_left, papers_right = set(data.loc[left_mask, "paper_id"]), set(data.loc[right_mask, "paper_id"])
        cross = (edges["paper_a"].isin(papers_left) & edges["paper_b"].isin(papers_right)) | (
            edges["paper_b"].isin(papers_left) & edges["paper_a"].isin(papers_right)
        )
        for language in SOURCES:
            near = edges["check"].isin(["near_abstract", "same_abstract_shingles"])
            row[f"near_abstract_{language}"] = int((cross & near & edges["language"].eq(language)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def verify_splits(reviews: pd.DataFrame, data: pd.DataFrame, audit: pd.DataFrame) -> None:
    if data["review_id"].tolist() != reviews.loc[reviews["label"].notna(), "review_id"].tolist():
        raise ValueError("Split files do not cover exactly the labeled review IDs")
    if data["split"].isna().any() or set(data["split"]) != set(SPLITS):
        raise ValueError("Incomplete split assignments")
    if (audit.drop(columns=["split_a", "split_b"]) != 0).any().any():
        raise ValueError("A paper or duplicate text crossed split boundaries")
