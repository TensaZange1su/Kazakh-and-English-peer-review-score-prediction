import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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
                app.button(key='run_prediction').click().run(timeout=180)
                self.assertFalse(app.exception)
                self.assertTrue(app.success)
                app.sidebar.radio[0].set_value('Prediction History').run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe[0].value), 1)
                app.sidebar.radio[0].set_value('Prediction').run()
                app.button(key='sample_kk').click().run()
                self.assertEqual(app.selectbox(key='input_language').value, 'kk')
                self.assertIn('Ғылыми', app.text_input(key='paper_title').value)
                app.toggle(key='compare_models').set_value(True).run()

                def fake_predict(service, text, language, *, title=''):
                    return dict(model_id=service.model_id, title=title, abstract=text, language=language,
                                predicted_label='Moderate', inference_ms=1.0, confidence=None)

                # Exercise comparison orchestration without downloading an encoder in the UI test.
                with patch('services.model_service.ModelService.predict', new=fake_predict):
                    app.button(key='run_prediction').click().run(timeout=180)
                self.assertFalse(app.exception)
                self.assertEqual(len(app.success), 2)
                app.text_area(key='paper_abstract').set_value('Updated abstract').run()
                self.assertTrue(app.warning)
                app.sidebar.radio[0].set_value('Prediction History').run()
                self.assertEqual(len(app.dataframe[0].value), 3)
                app.text_input[0].set_value('no matching title 123456').run()
                self.assertEqual(len(app.dataframe), 0)
                self.assertTrue(app.info)
                for page in ['Model Comparison', 'Analytics', 'Home']:
                    app.sidebar.radio[0].set_value(page).run()
                    self.assertFalse(app.exception)
            finally:
                os.environ.pop('APP_DB_PATH', None)


if __name__ == '__main__':
    unittest.main()
