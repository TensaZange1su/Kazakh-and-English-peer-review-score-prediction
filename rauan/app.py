"""Interactive bilingual research demo. Run with streamlit run app.py."""
import logging
import altair as alt
import pandas as pd
import streamlit as st
from database.repository import Repository
from services.model_service import ModelService, PredictionError, ROOT, available_models

st.set_page_config(page_title='Peer Review Lab', page_icon='📄', layout='wide')
st.markdown('''<style>
.block-container {max-width:1180px;padding-top:2rem;}
[data-testid="stMetric"] {border:1px solid #64748b40;border-radius:14px;padding:16px;}
.lab-eyebrow {color:#14b8a6;font-weight:700;letter-spacing:.16em;font-size:.8rem;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def repository():
    repo = Repository()
    repo.import_results()
    return repo

SAMPLES = {
    'en': ('Multilingual Classification of Scientific Abstracts',
           'This study compares TF-IDF features and Transformer embeddings for classifying scientific abstracts in English and Kazakh. We train logistic regression classifiers using a bilingual dataset and evaluate them on papers excluded from training. Performance is measured using accuracy and macro F1. The results highlight the challenges of limited training data, class imbalance, and differences between languages.'),
    'kk': ('Ғылыми аннотацияларды көптілді жіктеу',
           'Бұл зерттеуде ағылшын және қазақ тілдеріндегі ғылыми аннотацияларды жіктеу үшін TF-IDF белгілері мен Transformer модельдерінің векторлық көріністері салыстырылады. Екітілді деректер жиыны негізінде логистикалық регрессия модельдері оқытылады. Бағалау оқытуға енгізілмеген мақалаларда жүргізіледі. Модельдердің нәтижелері дәлдік және макро F1 көрсеткіштері арқылы өлшенеді. Зерттеу деректердің аздығы мен сыныптар теңгерімсіздігінің әсерін қарастырады.')
}


def example(language):
    st.session_state.input_language = language
    st.session_state.paper_title, st.session_state.paper_abstract = SAMPLES[language]
    st.session_state.pop('latest_run', None)


def clear_input():
    st.session_state.paper_title = ''
    st.session_state.paper_abstract = ''
    st.session_state.pop('latest_run', None)


def navigate(page):
    st.session_state.navigation = page


def label(model):
    return model.replace('e5_', 'E5 · ').replace('_', ' ').title()


def metric_chart(frame, x, y, *, percent=False):
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=5).encode(
        x=alt.X(x, title=x.replace('_', ' ').title(), scale=alt.Scale(domain=[0, 1]) if percent else alt.Undefined),
        y=alt.Y(y, sort='-x', title=None),
        color=alt.Color('family:N', scale=alt.Scale(domain=['TF-IDF', 'Transformer'], range=['#14b8a6', '#818cf8'])),
        tooltip=list(frame.columns)
    ).properties(height=max(180, len(frame) * 35)).interactive()
    st.altair_chart(chart, use_container_width=True)


repo = repository()
st.sidebar.title('📄 Peer Review Lab')
page = st.sidebar.radio('Navigate', ['Home', 'Prediction', 'Model Comparison', 'Analytics', 'Prediction History'], key='navigation')
st.sidebar.caption('English · Қазақша | TF-IDF · E5')
st.sidebar.divider()
st.sidebar.caption('A research playground for bilingual scientific abstracts.')
st.markdown('<div class="lab-eyebrow">EXPLORE · PREDICT · COMPARE</div>', unsafe_allow_html=True)

if page == 'Home':
    st.title('Two languages. A new perspective.')
    st.write('Explore how machine learning predicts reviewer score categories from scientific titles and abstracts.')
    a, b, c = st.columns(3)
    a.metric('Languages', '2', help='English and Kazakh')
    b.metric('Model configurations', '12')
    c.metric('Labeled bilingual review pairs', '385')
    st.write('')
    a, b = st.columns(2)
    with a.container(border=True):
        st.subheader('🧪 Try the prediction playground')
        st.write('Load a sample, edit its abstract, and see how different models respond.')
        st.button('Open playground →', on_click=navigate, args=('Prediction',), type='primary')
    with b.container(border=True):
        st.subheader('📊 Explore the evidence')
        st.write('Filter evaluation metrics and compare local inference speed across models.')
        st.button('Explore analytics →', on_click=navigate, args=('Analytics',))
    st.info('Predictions reflect reviewer labels in a small research dataset. They do not determine a paper’s objective quality.')
    with st.expander('How does it work?'):
        st.write('1. Choose English or Kazakh and enter a title and abstract.\n2. Select a TF-IDF or E5 model, or compare both.\n3. Inspect the predicted category and find the saved result in History.')
        st.caption('Submissions are stored in local SQLite and visible to everyone using this app. E5 inputs are truncated to 512 tokens. Confidence is unavailable because calibration has not been verified.')

elif page == 'Prediction':
    st.title('Prediction playground')
    st.write('Bring an abstract, or start with an example. Change the wording and explore the result.')
    a, b, c = st.columns(3)
    a.button('🇬🇧 Load English example', key='sample_en', on_click=example, args=('en',), use_container_width=True)
    b.button('🇰🇿 Қазақша мысал', key='sample_kk', on_click=example, args=('kk',), use_container_width=True)
    c.button('Clear input', on_click=clear_input, use_container_width=True)
    language = st.selectbox('Language', ['en', 'kk'], key='input_language', format_func=lambda x: {'en':'English','kk':'Қазақша'}[x])
    compare = st.toggle('Compare TF-IDF and Transformer side by side', key='compare_models')
    if compare:
        left, right = st.columns(2)
        model_id = left.selectbox('TF-IDF configuration', [m for m in available_models(language) if not m.startswith('e5_')], format_func=label)
        second_model = right.selectbox('Transformer configuration', [m for m in available_models(language) if m.startswith('e5_')], format_func=label)
        selected = [model_id, second_model]
    else:
        family = st.radio('Model family', ['TF-IDF', 'Transformer'], horizontal=True)
        models = [m for m in available_models(language) if m.startswith('e5_') == (family == 'Transformer')]
        model_id = st.selectbox('Model configuration', models, format_func=label)
        selected = [model_id]
    title = st.text_input('Title', key='paper_title', max_chars=5000, placeholder='Give your paper a title…')
    abstract = st.text_area('Abstract', key='paper_abstract', height=230, max_chars=95000, placeholder='Describe the research question, methods, and findings…')
    a, b = st.columns(2)
    a.caption(f'{len(abstract.split())} words · {len(abstract):,} characters')
    b.caption('E5 uses the first 512 tokens; first use may take longer to load.')
    if st.button('Compare models' if compare else 'Predict', key='run_prediction', type='primary'):
        results = []
        progress = st.progress(0, text='Preparing your experiment…')
        for index, model in enumerate(selected):
            try:
                with st.spinner(f'Running {label(model)}…'):
                    result = ModelService(model).predict(abstract, language, title=title)
                    result['id'] = repo.save_prediction(result)
                results.append(result)
            except (ValueError, PredictionError) as exc:
                st.error(str(exc))
            except Exception:
                logging.exception('Prediction persistence failed')
                st.error('The prediction could not be saved. Check database permissions and disk space before retrying.')
            progress.progress((index + 1) / len(selected), text=f'Completed {index + 1} of {len(selected)} models')
        progress.empty()
        st.session_state.latest_run = dict(results=results, signature=(title, abstract, language, tuple(selected)))
    run = st.session_state.get('latest_run')
    if run and run['results']:
        stale = run['signature'] != (title, abstract, language, tuple(selected))
        if stale:
            st.warning('Input or model selection changed. Run prediction again to update the results below.')
        st.subheader('Your experiment results')
        for column, result in zip(st.columns(len(run['results'])), run['results']):
            with column.container(border=True):
                st.caption(label(result['model_id']))
                icon = {'Low':'🔵', 'Moderate':'🟠', 'High':'🟢'}[result['predicted_label']]
                st.subheader(f'{icon} {result["predicted_label"]}')
                st.success(f'Predicted review score: {result["predicted_label"]}')
                st.metric('Warm inference', f'{result["inference_ms"]:.1f} ms')
                st.caption(f'Saved to history · #{result["id"]} · excludes model loading')
        if len(run['results']) == 2:
            agreed = run['results'][0]['predicted_label'] == run['results'][1]['predicted_label']
            st.info('🤝 Both models agree on this input.' if agreed else '🔎 The models disagree. Explore their test-set metrics in Analytics; agreement alone does not establish correctness.')
        st.caption('These are reviewer score categories, not a quality verdict. Confidence unavailable: calibration has not been verified.')
        st.button('View saved predictions →', on_click=navigate, args=('Prediction History',))

elif page == 'Model Comparison':
    st.title('Model comparison')
    st.write('Explore speed and storage, then inspect the original measurements.')
    rows = repo.experiments('application_benchmark.csv')
    if rows:
        frame = pd.DataFrame(rows)
        frame['family'] = frame.model.map(lambda m:'Transformer' if m.startswith('e5_') else 'TF-IDF')
        chosen = st.multiselect('Model families', ['TF-IDF','Transformer'], default=['TF-IDF','Transformer'])
        frame = frame[frame.family.isin(chosen)]
        metric = st.selectbox('Compare by', ['inference_1000_seconds', 'texts_per_second', 'classifier_artifact_mib'])
        frame[metric] = pd.to_numeric(frame[metric])
        if len(frame):
            metric_chart(frame[['model','family',metric]], metric, 'model')
        else:
            st.info('Choose a model family to see measurements.')
        st.caption('Local CPU measurements. Lower time and storage are better; higher throughput is better. Classifier storage excludes the shared E5 encoder.')
    st.caption('Historical measurements below use different hardware and timing protocols. E5 timing includes embedding generation. Source MB columns use MiB (1024² bytes).')
    for source, name in [('tfidf_benchmark.csv','TF-IDF benchmark'), ('transformer_benchmark.csv','E5 benchmark'),
                         ('transformer_model_size.csv','E5 storage and parameters'), ('application_benchmark.csv','Local application benchmark')]:
        with st.expander(name):
            rows = repo.experiments(source)
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info('No local benchmark yet. Run python -m scripts.benchmark.')
    st.caption('E5 classifier joblib files exclude encoder weights. Repository snapshot size and loaded weight memory are separate measures.')

elif page == 'Analytics':
    st.title('Explore evaluation results')
    language = st.selectbox('Evaluation scope', ['english', 'kazakh', 'combined'])
    metric = st.selectbox('Metric', ['macro_f1','accuracy','precision_macro','recall_macro','weighted_f1'])
    rows = []
    for source in ('tfidf_detailed_metrics.csv', 'transformer_metrics.csv'):
        for row in repo.experiments(source):
            if row['evaluation_language'] == language:
                rows.append({'model': row['model_id'], 'family':'Transformer' if row['model_id'].startswith('e5_') else 'TF-IDF',
                             **{key: float(row.get(key, row.get('test_' + key))) for key in ('accuracy','precision_macro','recall_macro','macro_f1','weighted_f1')}})
    if rows:
        frame = pd.DataFrame(rows)
        selected = st.multiselect('Models to display', frame.model.tolist(), default=frame.model.tolist())
        frame = frame[frame.model.isin(selected)]
        if len(frame):
            metric_chart(frame[['model','family',metric]], metric, 'model', percent=True)
            st.dataframe(frame.set_index('model'), use_container_width=True)
        else:
            st.info('Choose at least one model to explore the results.')
    st.caption('Held-out test metrics. Combined results pool both languages. Models were selected using validation data; the chart is exploratory.')
    images = sorted((ROOT / 'results/confusion_matrices').glob(f'*_{language}.png'))
    if language == 'english':
        images += list((ROOT / 'results/error_analysis').glob('*confusion_matrix.png'))
    with st.expander('Inspect confusion matrices'):
        if images:
            image = st.selectbox('Confusion matrix', images, format_func=lambda p:p.stem)
            st.image(str(image), width=600)

else:
    st.title('Your experiment history')
    language = st.selectbox('Language filter', [None, 'en', 'kk'], format_func=lambda x: x or 'All')
    rows = repo.history(language)
    st.caption('Latest 500 predictions · timestamps in UTC · inputs are stored locally')
    if not rows:
        st.info('No predictions yet. Submit an abstract on the Prediction page.')
    else:
        frame = pd.DataFrame(rows)
        search = st.text_input('Search title or abstract', placeholder='Find a previous experiment…')
        model_filter = st.multiselect('Filter models', sorted(frame.model_id.unique()), default=[])
        score_filter = st.multiselect('Filter scores', ['Low','Moderate','High'], default=[])
        if search:
            frame = frame[frame.title.str.contains(search, case=False, regex=False) | frame.abstract.str.contains(search, case=False, regex=False)]
        if model_filter:
            frame = frame[frame.model_id.isin(model_filter)]
        if score_filter:
            frame = frame[frame.predicted_label.isin(score_filter)]
        st.metric('Matching predictions', len(frame))
        if frame.empty:
            st.info('No matches. Try another search or clear the filters.')
        else:
            st.dataframe(frame.drop(columns=['abstract', 'confidence']), hide_index=True, use_container_width=True)
            record = st.selectbox('Inspect prediction', frame.to_dict('records'), format_func=lambda r:f'#{r["id"]} — {r["title"] or "Untitled"}')
            with st.expander('Full submitted text', expanded=True):
                st.text(record['title'])
                st.text(record['abstract'])
            st.download_button('Export filtered history', frame.to_csv(index=False), 'prediction_history.csv', 'text/csv')
