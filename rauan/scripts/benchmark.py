import argparse
import csv
import platform
import os
from pathlib import Path
from statistics import median
from time import perf_counter

from database.repository import Repository
from services.model_service import ROOT, APP_ROOT, MODEL_IDS, load_encoder, load_model, prepare_text


def synchronize(encoder):
    if encoder is not None and encoder.device.type == 'cuda':
        import torch
        torch.cuda.synchronize()


def benchmark(model_id, repeats):
    language = 'kk' if 'kazakh' in model_id else 'en'
    with (ROOT / 'data/splits/test.csv').open() as stream:
        source = list(csv.DictReader(stream))
    pairs = [(r[f'title_{lang}'], r[f'abstract_{lang}'])
             for lang in (('en', 'kk') if 'multilingual' in model_id else (language,)) for r in source]
    pairs = (pairs * (1000 // len(pairs) + 1))[:1000]
    texts = [prepare_text(t, a, model_id) for t, a in pairs]
    model = load_model(model_id)
    encoder = load_encoder() if model_id.startswith('e5_') else None

    def run(batch):
        features = encoder.encode(batch, batch_size=32, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False) if encoder else batch
        return model.predict(features)

    run(texts[:32])
    timings = []
    for _ in range(repeats):
        synchronize(encoder)
        start = perf_counter()
        labels = run(texts)
        synchronize(encoder)
        timings.append(perf_counter() - start)
    if len(labels) != 1000:
        raise RuntimeError('Incomplete inference batch')
    folder = 'transformer' if encoder else 'tfidf'
    row = dict(model=model_id, language='combined' if 'multilingual' in model_id else language,
               texts=1000, repeats=repeats, batch_size=32 if encoder else 1000,
               inference_1000_seconds=median(timings),
               texts_per_second=1000 / median(timings),
               classifier_artifact_mib=(ROOT / 'models' / folder / f'{model_id}.joblib').stat().st_size / 1024**2,
               encoder_parameters='', encoder_weight_mib='', encoder_directory_mib='',
               device=str(encoder.device) if encoder else 'cpu', hardware=platform.platform())
    if encoder:
        row['encoder_parameters'] = sum(p.numel() for p in encoder.parameters())
        row['encoder_weight_mib'] = sum(p.numel() * p.element_size() for p in encoder.parameters()) / 1024**2
        # Count the active encoder snapshot, deduplicating linked cache files.
        path = getattr(encoder[0].auto_model.config, '_name_or_path', '')
        if not Path(path).is_dir():
            from huggingface_hub import try_to_load_from_cache
            config = try_to_load_from_cache(os.getenv('E5_MODEL_PATH', 'intfloat/multilingual-e5-small'), 'config.json')
            if isinstance(config, str):
                path = str(Path(config).parent)
        if Path(path).is_dir():
            files = {p.resolve() for p in Path(path).rglob('*') if p.is_file()}
            row['encoder_directory_mib'] = sum(p.stat().st_size for p in files) / 1024**2
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    output = APP_ROOT / 'results/application_benchmark.csv'
    existing = {}
    if output.exists():
        with output.open() as stream:
            existing = {r['model']: r for r in csv.DictReader(stream)}
    for model_id in args.models:
        row = benchmark(model_id, args.repeats)
        existing[model_id] = row
        # Persist each completed model so a later encoder failure does not lose TF-IDF results.
        with output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(existing.values())
        print(f'{model_id}: {row["inference_1000_seconds"]:.3f}s / 1000 texts', flush=True)
    Repository().import_results()


if __name__ == '__main__':
    main()
