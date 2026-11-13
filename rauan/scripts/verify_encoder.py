import csv
import numpy as np
from services.model_service import ROOT, load_encoder, load_model, prepare_text


def main():
    with (ROOT / 'data/splits/test.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    encoder = load_encoder()
    for language, name in [('en', 'english'), ('kk', 'kazakh')]:
        texts = [prepare_text(r[f'title_{language}'], r[f'abstract_{language}'], f'e5_{name}') for r in rows]
        actual = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        reference = np.load(ROOT / f'models/transformer/embeddings/test_{language}.npy')
        np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-3)
        for model_id in [f'e5_{name}', 'e5_multilingual']:
            model = load_model(model_id)
            np.testing.assert_array_equal(model.predict(actual), model.predict(reference))
        print(f'{language}: embeddings and classifier predictions match', flush=True)


if __name__ == '__main__':
    main()
