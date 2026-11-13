from functools import lru_cache
from pathlib import Path
import os
from time import perf_counter

APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = APP_ROOT.parent
LANGUAGES = {'en': 'english', 'kk': 'kazakh'}
MODEL_IDS = tuple(f'{lang}_{feature}' for lang in ('english', 'kazakh', 'multilingual')
                  for feature in ('word', 'char', 'combined')) + tuple(
                      f'e5_{lang}' for lang in ('english', 'kazakh', 'multilingual'))


class PredictionError(RuntimeError):
    pass


def available_models(language):
    if not isinstance(language, str) or language not in LANGUAGES:
        raise ValueError('Language must be en or kk.')
    return [m for m in MODEL_IDS if LANGUAGES[language] in m or 'multilingual' in m]


def validate(title, abstract, language, model_id):
    if model_id not in available_models(language):
        raise ValueError('Model is unavailable for the selected language.')
    if not isinstance(title, str) or not isinstance(abstract, str):
        raise ValueError('Title and abstract must be strings.')
    if not abstract.strip():
        raise ValueError('Enter a nonempty abstract.')
    if len(title) + len(abstract) > 100_000:
        raise ValueError('Title and abstract must total at most 100,000 characters.')


def prepare_text(title, abstract, model_id):
    if model_id.startswith('e5_'):
        return f'query: Title: {title} Abstract: {abstract}'
    return f'{title} [SEP] {abstract}'


@lru_cache(maxsize=12)
def load_model(model_id):
    if model_id not in MODEL_IDS:
        raise ValueError('Unknown model.')
    import joblib
    folder = 'transformer' if model_id.startswith('e5_') else 'tfidf'
    return joblib.load(ROOT / 'models' / folder / f'{model_id}.joblib')


@lru_cache(maxsize=1)
def load_encoder():
    from sentence_transformers import SentenceTransformer
    local = APP_ROOT / 'models/transformer/encoder'
    source = os.getenv('E5_MODEL_PATH') or (str(local) if local.is_dir() else 'intfloat/multilingual-e5-small')
    encoder = SentenceTransformer(source, device=os.getenv('E5_DEVICE', 'cpu'))
    encoder.max_seq_length = 512
    return encoder


class ModelService:
    def __init__(self, model_id=None):
        self.model_id = model_id

    def predict(self, text, language, *, title=''):
        """Predict an abstract; title is optional. No database side effects."""
        available_models(language)
        model_id = self.model_id or f'{LANGUAGES[language]}_char'
        validate(title, text, language, model_id)
        try:
            model = load_model(model_id)
            encoder = load_encoder() if model_id.startswith('e5_') else None
            start = perf_counter()  # warm inference; excludes artifact loading
            prepared = prepare_text(title, text, model_id)
            features = encoder.encode([prepared], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False) if encoder else [prepared]
            label = str(model.predict(features)[0])
            if label not in ('Low', 'Moderate', 'High'):
                raise ValueError('Model returned an unknown label.')
            return dict(model_id=model_id, language=language, title=title, abstract=text,
                        predicted_label=label, inference_ms=(perf_counter() - start) * 1000,
                        confidence=None)
        except Exception as exc:
            raise PredictionError(f'Could not run {model_id}. Check dependencies, model files, and E5 cache. {exc}') from exc


def predict(text, language):
    return ModelService().predict(text, language)
