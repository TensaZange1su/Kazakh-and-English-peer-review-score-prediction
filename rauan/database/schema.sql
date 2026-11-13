CREATE TABLE IF NOT EXISTS experiments (
 id INTEGER PRIMARY KEY, model_id TEXT NOT NULL, language TEXT NOT NULL,
 source TEXT NOT NULL, metrics_json TEXT NOT NULL,
 imported_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
 UNIQUE(model_id, language, source)
);
CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL,
 language TEXT NOT NULL CHECK(language IN ('en','kk')), model_id TEXT NOT NULL,
 predicted_label TEXT NOT NULL CHECK(predicted_label IN ('Low','Moderate','High')),
 inference_ms REAL NOT NULL CHECK(inference_ms >= 0), confidence REAL,
 created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS predictions_created ON predictions(created_at);
