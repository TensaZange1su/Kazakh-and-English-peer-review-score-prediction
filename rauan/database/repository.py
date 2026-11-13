from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import sqlite3

APP_ROOT = Path(__file__).resolve().parents[1]
ROOT = APP_ROOT.parent


class Repository:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv('APP_DB_PATH', str(APP_ROOT / 'data' / 'application.sqlite3')))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript((APP_ROOT / 'database/schema.sql').read_text())

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_prediction(self, result):
        keys = ('title', 'abstract', 'language', 'model_id', 'predicted_label', 'inference_ms', 'confidence')
        with self.connect() as db:
            return db.execute(f'INSERT INTO predictions ({",".join(keys)}) VALUES (?,?,?,?,?,?,?)',
                              [result[k] for k in keys]).lastrowid

    def history(self, language=None, limit=500):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                'SELECT * FROM predictions WHERE (? IS NULL OR language=?) ORDER BY id DESC LIMIT ?',
                (language, language, limit))]

    def import_results(self):
        sources = ('tfidf_detailed_metrics.csv', 'transformer_metrics.csv', 'tfidf_benchmark.csv',
                   'transformer_benchmark.csv', 'transformer_model_size.csv', 'application_benchmark.csv')
        with self.connect() as db:
            for source in sources:
                path = (APP_ROOT if source == 'application_benchmark.csv' else ROOT) / 'results' / source
                if not path.exists():
                    continue
                with path.open(encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    model = row.get('experiment') or row['model']
                    language = row.get('language') or ('combined' if 'multilingual' in model else
                                                     'kazakh' if 'kazakh' in model else 'english')
                    db.execute('''INSERT INTO experiments(model_id,language,source,metrics_json)
                        VALUES (?,?,?,?) ON CONFLICT(model_id,language,source) DO UPDATE SET
                        metrics_json=excluded.metrics_json''', (model, language, source, json.dumps(row)))

    def experiments(self, source=None):
        with self.connect() as db:
            return [dict(model_id=r['model_id'], evaluation_language=r['language'], source=r['source'],
                         **json.loads(r['metrics_json'])) for r in db.execute(
                             'SELECT * FROM experiments WHERE (? IS NULL OR source=?)', (source, source))]
