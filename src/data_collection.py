"""Acquire and validate the two pinned SIST releases."""

from hashlib import sha256
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "merged_dataset.csv"
SOURCES = {
    "en": (
        "sist-english-corpus", "bef7222ed5e05cb84bed63e31c28a77c40d389a6",
        "Articles_SIST_clean.csv", "4ccc35366b6cf6174221493c2f2070656faaa666a1b01f05ad0baafd703b9adb",
    ),
    "kk": (
        "sist-kazakh-corpus", "d21d147364c2a35c9eba2a3791da793c9e06d95e",
        "Articles_SIST_clean_kaz.csv", "84f19e4926046bb24d69aa3bfc46edefb87cdd8c5f08829df74a264412af8a18",
    ),
}
AUTHOR = "https://huggingface.co/datasets/Arailym-tleubayeva"
LABELS = ["Low", "Moderate", "High"]
SCORE_MAP = dict(zip(["Төмен", "Орташа", "Жоғары"], LABELS))
TEXT_FIELDS = ["Title", "Abstract", "normalized_title"]
EXPECTED_COLUMNS = ["Unnamed: 0", *TEXT_FIELDS, "Keywords", "CitationCount", "Review_score", "Review_comment"]


def normalize(text: str) -> str:
    """Normalize case, Unicode and spacing for matching only."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def source_url(language: str) -> str:
    repository, revision, filename, _ = SOURCES[language]
    return f"{AUTHOR}/{repository}/resolve/{revision}/{filename}"


def load_sources() -> dict[str, pd.DataFrame]:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = {}
    for language, (_, _, _, expected_hash) in SOURCES.items():
        path = RAW / f"sist_{language}.csv"
        if path.is_file():
            content = path.read_bytes()
        else:
            with urlopen(source_url(language), timeout=60) as response:
                content = response.read()
        if sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"Unexpected SHA-256 for {path.name}. Use the pinned source at {source_url(language)}")
        if not path.is_file():
            path.write_bytes(content)
        frame = pd.read_csv(path, encoding="utf-8")
        if set(frame.columns) != set(EXPECTED_COLUMNS) or len(frame) != 415:
            raise ValueError(f"Unexpected schema or row count in {path.name}")
        if frame["Unnamed: 0"].isna().any() or not frame["Unnamed: 0"].is_unique:
            raise ValueError(f"Missing or repeated source row IDs in {path.name}")
        for field in [*TEXT_FIELDS, "Keywords", "Review_score", "Review_comment"]:
            frame[field] = frame[field].str.strip().replace("", pd.NA)
        for field in TEXT_FIELDS:
            if frame[field].isna().any():
                raise ValueError(f"Missing title or abstract content in {path.name}: {field}")
        frames[language] = frame
    return frames


def stable_id(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


def pair_reviews(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    en, kk = (frames[lang].rename(columns={"Unnamed: 0": "review_id"}) for lang in ("en", "kk"))
    pair = en.merge(kk, on="review_id", suffixes=("_en", "_kk"), validate="one_to_one", how="outer", indicator=True)
    if not pair["_merge"].eq("both").all():
        raise ValueError("The published source row ID sets differ")
    english = pair["Review_score_en"]
    kazakh = pair["Review_score_kk"]
    if set(english.dropna()) != set(LABELS) or set(kazakh.dropna()) != set(SCORE_MAP):
        raise ValueError("Unexpected review score vocabulary")
    if not english.fillna("").eq(kazakh.map(SCORE_MAP).fillna("")).all():
        raise ValueError("Paired review scores or their missingness differ")
    if not pair["CitationCount_en"].fillna(-1).eq(pair["CitationCount_kk"].fillna(-1)).all():
        raise ValueError("Paired citation counts differ")
    for language in SOURCES:
        field = f"normalized_title_{language}"
        pair[field] = pair[field].map(normalize)
    title_pairs = pair[["normalized_title_en", "normalized_title_kk"]].drop_duplicates()
    if any(not title_pairs[field].is_unique for field in title_pairs):
        raise ValueError("Normalized English and Kazakh titles do not map one-to-one")

    pair["paper_id"] = pair["normalized_title_en"].map(lambda title: "p_" + stable_id(title))
    if pair["paper_id"].nunique() != len(title_pairs):
        raise ValueError("Paper ID hash collision")
    pair = pair.rename(columns={"Review_score_en": "label", **{
        f"{field}_{lang}": f"{field.lower()}_{lang}"
        for field in ("Title", "Abstract") for lang in SOURCES
    }})
    columns = ["review_id", "paper_id", "label", "title_en", "abstract_en", "title_kk", "abstract_kk",
               "normalized_title_en", "normalized_title_kk"]
    return pair[columns].sort_values("review_id").reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, encoding="utf-8", index=False, lineterminator="\n")


def token_count(text: str) -> int:
    """Approximate words via Unicode word tokens, including Kazakh letters."""
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
