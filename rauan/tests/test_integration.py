import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from database.repository import Repository
from services.model_service import ModelService, ROOT, available_models, load_model, prepare_text


class IntegrationTests(unittest.TestCase):
    def test_invalid_inputs(self):
        for text, lang in [('', 'en'), ('  ', 'kk'), (None, 'en'), ('abstract', 'ru'), ('abstract', []), ('abstract', None)]:
            with self.assertRaises(ValueError):
                ModelService().predict(text, lang)
        with self.assertRaises(ValueError):
            ModelService('english_char').predict('text', 'kk')

    def test_training_format(self):
        self.assertEqual(prepare_text('Title', 'Abstract', 'english_char'), 'Title [SEP] Abstract')
        self.assertEqual(prepare_text('Title', 'Abstract', 'e5_english'), 'query: Title: Title Abstract: Abstract')

    def test_database_roundtrip_and_import(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Repository(Path(directory) / 'app.sqlite3')
            repo.import_results()
            before = len(repo.experiments())
            repo.import_results()
            self.assertEqual(len(repo.experiments()), before)
            result = dict(title="Қазақша ' ; DROP TABLE predictions;", abstract='Сынақ мәтіні',
                          language='kk', model_id='kazakh_char', predicted_label='Moderate',
                          inference_ms=1.2, confidence=None)
            repo.save_prediction(result)
            repo.save_prediction(result)
            reopened = Repository(repo.path)
            self.assertEqual(len(reopened.history('kk')), 2)
            self.assertEqual(reopened.history()[0]['title'], result['title'])
            self.assertEqual(reopened.history('en'), [])

    def test_all_tfidf_predictions_match_handoff(self):
        import numpy as np
        with (ROOT / 'data/splits/test.csv').open() as stream:
            papers = {r['review_id']: r for r in csv.DictReader(stream)}
        with (ROOT / 'results/tfidf_predictions/all_predictions.csv').open() as stream:
            predictions = list(csv.DictReader(stream))
        for model_id in available_models('en') + available_models('kk'):
            if model_id.startswith('e5_'):
                continue
            for language, label in [('en', 'english'), ('kk', 'kazakh')]:
                if model_id not in available_models(language):
                    continue
                rows = [r for r in predictions if r['experiment'] == model_id and r['language'] == label]
                self.assertEqual(len(rows), 58)
                texts = [prepare_text(papers[r['review_id']][f'title_{language}'],
                                      papers[r['review_id']][f'abstract_{language}'], model_id) for r in rows]
                np.testing.assert_array_equal(load_model(model_id).predict(texts), [r['predicted_label'] for r in rows])
                paper = papers[rows[0]['review_id']]
                result = ModelService(model_id).predict(paper[f'abstract_{language}'], language,
                                                       title=paper[f'title_{language}'])
                self.assertEqual(result['predicted_label'], rows[0]['predicted_label'])
                self.assertIsNone(result['confidence'])

    def test_transformer_handoff_metrics(self):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        with (ROOT / 'data/splits/test.csv').open() as stream:
            labels = [r['label'] for r in csv.DictReader(stream)]
        with (ROOT / 'results/transformer_metrics.csv').open() as stream:
            metrics = list(csv.DictReader(stream))
        for row in metrics:
            name = row['model'].removeprefix('e5_')
            suffix = {'english': 'en', 'kazakh': 'kk', 'multilingual': 'multi'}[name]
            features = np.load(ROOT / f'models/transformer/embeddings/test_{suffix}.npy')
            truth = labels * (2 if suffix == 'multi' else 1)
            predictions = load_model(row['model']).predict(features)
            self.assertAlmostEqual(accuracy_score(truth, predictions), float(row['test_accuracy']))
            self.assertAlmostEqual(f1_score(truth, predictions, average='macro'), float(row['test_macro_f1']))

    def test_e5_adapter_with_reference_embedding(self):
        """Real saved heads + held-out embeddings; encoder formatting checked separately."""
        import numpy as np
        for language, name in [('en', 'english'), ('kk', 'kazakh')]:
            embedding = np.load(ROOT / f'models/transformer/embeddings/test_{language}.npy')[:1]
            with (ROOT / 'data/splits/test.csv').open() as stream:
                paper = next(csv.DictReader(stream))
            for model_id in (f'e5_{name}', 'e5_multilingual'):
                class Encoder:
                    def encode(self, texts, **kwargs):
                        self_input = prepare_text(paper[f'title_{language}'], paper[f'abstract_{language}'], model_id)
                        if texts != [self_input] or not kwargs['normalize_embeddings']:
                            raise AssertionError('Invalid E5 preprocessing')
                        return embedding
                with patch('services.model_service.load_encoder', return_value=Encoder()):
                    result = ModelService(model_id).predict(paper[f'abstract_{language}'], language,
                                                           title=paper[f'title_{language}'])
                self.assertEqual(result['predicted_label'], load_model(model_id).predict(embedding)[0])


if __name__ == '__main__':
    unittest.main()
