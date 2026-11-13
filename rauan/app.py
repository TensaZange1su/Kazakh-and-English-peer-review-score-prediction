import logging
import pandas as pd
import streamlit as st
from database.repository import Repository
from services.model_service import ModelService, PredictionError, ROOT, available_models

st.set_page_config(page_title='Peer Review Lab', page_icon='📄', layout='wide')

@st.cache_resource
def repository():
    repo = Repository()
    repo.import_results()
    return repo

repo = repository()
st.sidebar.title('Peer Review Lab')
page = st.sidebar.radio('Navigate', ['Home', 'Prediction', 'Model Comparison', 'Analytics', 'Prediction History'])
st.sidebar.caption('English · Қазақша | TF-IDF · E5')

if page == 'Home':
    st.title('Peer-review score prediction')
    st.write('Explore how academic titles and abstracts relate to Low, Moderate, and High review scores in English and Kazakh.')
    st.info('Predictions reflect reviewer labels in a small research dataset. They do not determine a paper’s objective quality.')
    st.write('Choose Prediction to submit a paper, Model Comparison to compare experiments, or Analytics to explore evaluation results.')
    st.caption('385 labeled bilingual review pairs · paper-level train/validation/test splits · 12 trained model configurations')
    st.write('Submitted titles and abstracts are saved in the local SQLite database. Use a persistent disk when deploying.')

elif page == 'Prediction':
    st.title('Predict a review score')
    language = st.selectbox('Language', ['en', 'kk'], format_func=lambda x: {'en':'English','kk':'Қазақша'}[x])
    family = st.radio('Model family', ['TF-IDF', 'Transformer'], horizontal=True)
    models = [m for m in available_models(language) if m.startswith('e5_') == (family == 'Transformer')]
    model_id = st.selectbox('Model configuration', models)
    st.caption('E5 loads its encoder and tokenizer on first use; inputs are truncated to 512 tokens.')
    with st.form('prediction'):
        title = st.text_input('Title', max_chars=5000)
        abstract = st.text_area('Abstract', height=240, max_chars=95000)
        submitted = st.form_submit_button('Predict', type='primary')
    if submitted:
        try:
            with st.spinner('Running prediction…'):
                result = ModelService(model_id).predict(abstract, language, title=title)
            record_id = repo.save_prediction(result)
            st.success(f'Predicted review score: {result["predicted_label"]}')
            st.metric('Inference time (excludes model loading)', f'{result["inference_ms"]:.1f} ms')
            st.caption(f'Saved prediction #{record_id}. Confidence unavailable: calibration has not been verified for these artifacts.')
        except (ValueError, PredictionError) as exc:
            st.error(str(exc))
        except Exception:
            logging.exception('Prediction persistence failed')
            st.error('The prediction could not be saved. Check database permissions and disk space before retrying.')

elif page == 'Model Comparison':
    st.title('TF-IDF vs. Transformer')
    st.caption('Imported Participant 2 measurements; hardware and timing protocols differ. E5 timing includes embedding generation. MB columns in source files use MiB (1024² bytes).')
    for source, name in [('tfidf_benchmark.csv','TF-IDF benchmark'), ('transformer_benchmark.csv','E5 benchmark'),
                         ('transformer_model_size.csv','E5 storage and parameters'), ('application_benchmark.csv','Local application benchmark')]:
        st.subheader(name)
        rows = repo.experiments(source)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info('No local benchmark yet. Run python -m scripts.benchmark.')
    st.caption('E5 classifier joblib files exclude encoder weights. Repository snapshot size includes additional files and weight formats; loaded weight size is a separate measure.')

elif page == 'Analytics':
    st.title('Evaluation analytics')
    language = st.selectbox('Evaluation scope', ['english', 'kazakh', 'combined'])
    rows = []
    for source in ('tfidf_detailed_metrics.csv', 'transformer_metrics.csv'):
        for row in repo.experiments(source):
            if row['evaluation_language'] == language:
                rows.append({'model': row['model_id'], **{key: float(row.get(key, row.get('test_' + key)))
                             for key in ('accuracy','precision_macro','recall_macro','macro_f1','weighted_f1')}})
    if rows:
        frame = pd.DataFrame(rows).set_index('model')
        st.bar_chart(frame[['accuracy', 'macro_f1']])
        st.dataframe(frame, use_container_width=True)
    st.caption('Test-set metrics. Combined multilingual scores pool both languages; they are not language-specific results. Model choice was made on validation data.')
    images = sorted((ROOT / 'results/confusion_matrices').glob(f'*_{language}.png'))
    if language == 'english':
        images += list((ROOT / 'results/error_analysis').glob('*confusion_matrix.png'))
    if images:
        image = st.selectbox('Confusion matrix', images, format_func=lambda p:p.stem)
        st.image(str(image), width=600)

else:
    st.title('Prediction history')
    language = st.selectbox('Language filter', [None, 'en', 'kk'], format_func=lambda x: x or 'All')
    rows = repo.history(language)
    st.caption('Latest 500 predictions · timestamps in UTC · inputs are stored locally')
    if not rows:
        st.info('No predictions yet. Submit an abstract on the Prediction page.')
    else:
        frame = pd.DataFrame(rows)
        st.dataframe(frame.drop(columns=['abstract', 'confidence']), hide_index=True, use_container_width=True)
        record = st.selectbox('Inspect prediction', rows, format_func=lambda r:f'#{r["id"]} — {r["title"] or "Untitled"}')
        st.text(record['title'])
        st.text(record['abstract'])
        st.download_button('Export visible history', frame.to_csv(index=False), 'prediction_history.csv', 'text/csv')
