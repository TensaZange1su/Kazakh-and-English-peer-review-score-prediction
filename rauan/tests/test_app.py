import os
from pathlib import Path
import tempfile
import unittest
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_prediction_history_and_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ['APP_DB_PATH'] = str(Path(directory) / 'app.sqlite3')
            try:
                app = AppTest.from_file('app.py').run(timeout=180)
                self.assertFalse(app.exception)
                app.sidebar.radio[0].set_value('Prediction').run()
                app.text_input[0].set_value('Academic machine learning')
                app.text_area[0].set_value('We evaluate multilingual classification methods on scientific abstracts.')
                app.button[0].click().run(timeout=180)
                self.assertFalse(app.exception)
                self.assertTrue(app.success)
                app.sidebar.radio[0].set_value('Prediction History').run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe[0].value), 1)
                for page in ['Model Comparison', 'Analytics', 'Home']:
                    app.sidebar.radio[0].set_value(page).run()
                    self.assertFalse(app.exception)
            finally:
                os.environ.pop('APP_DB_PATH', None)


if __name__ == '__main__':
    unittest.main()
