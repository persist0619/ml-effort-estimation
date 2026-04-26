import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.models import load_data, train_all_models, predict_effort, SHORT_NAMES
from core.database import save_estimation
from core.report import generate_pdf

st.set_page_config(page_title='工作量预测', layout='wide')
st.title('工作量预测')

DEV_MODE_OPTIONS = {'有机型': 0, '半分离型': 1, '嵌入型': 2}
LANG_TYPE_OPTIONS = {'低级语言': 0, '高级语言': 1, '超高级语言': 2}
RECOMMENDED_MODEL = 'SVM (RBF)'


def ensure_models():
    if 'trained_models' not in st.session_state:
        with st.spinner('首次使用，正在训练模型...'):
            _, X, y = load_data()
            trained, scaler = train_all_models(X, y)
            st.session_state['trained_models'] = trained
            st.session_state['scaler'] = scaler


def render_input_form(prefix, default_name=''):
    name = st.text_input('项目名称', value=default_name, key=f'{prefix}_name')
    c1, c2 = st.columns(2)
    with c1:
        fp = st.slider('功能点计数', 50, 1500, 300, key=f'{prefix}_fp')
        complexity = st.slider('项目复杂度', 1.0, 5.0, 3.0, 0.1, key=f'{prefix}_cx')
        code_size = st.slider('代码规模 (KLoC)', 5, 500, 50, key=f'{prefix}_cs')
        dev_mode_label = st.selectbox('开发模式', list(DEV_MODE_OPTIONS.keys()), key=f'{prefix}_dm')
    with c2:
        team_exp = st.slider('团队经验', 1.0, 5.0, 3.5, 0.1, key=f'{prefix}_te')
        tool_mat = st.slider('工具成熟度', 1.0, 5.0, 3.0, 0.1, key=f'{prefix}_tm')
        lang_label = st.selectbox('语言类型', list(LANG_TYPE_OPTIONS.keys()), key=f'{prefix}_lt')

    return {
        'project_name': name,
        'function_points': fp,
        'project_complexity': complexity,
        'code_size_kloc': code_size,
        'team_experience': team_exp,
        'tool_maturity': tool_mat,
        'dev_mode': DEV_MODE_OPTIONS[dev_mode_label],
        'language_type': LANG_TYPE_OPTIONS[lang_label],
    }


def render_report(params, predictions):
    st.markdown(f"#### {params['project_name']}")

    st.markdown('**项目参数**')
    param_df = pd.DataFrame([{
        '功能点': params['function_points'],
        '复杂度': params['project_complexity'],
        '代码规模(KLoC)': params['code_size_kloc'],
        '团队经验': params['team_experience'],
        '工具成熟度': params['tool_maturity'],
        '开发模式': params['dev_mode'],
        '语言类型': params['language_type'],
    }])
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    st.markdown('**模型预测结果**')
    pred_rows = []
    for name, val in predictions.items():
        pred_rows.append({
            '模型': name,
            '预测工作量 (人月)': f'{val:.2f}',
            '推荐': '★' if name == RECOMMENDED_MODEL else '',
        })
    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    rec_effort = predictions[RECOMMENDED_MODEL]
    st.markdown('**工期与人力换算**')
    schedule = []
    for size in [1, 2, 3, 5]:
        schedule.append({'团队规模 (人)': size, '预计工期 (月)': f'{rec_effort / size:.1f}'})
    st.dataframe(pd.DataFrame(schedule), use_container_width=True, hide_index=True)

    model_names = list(predictions.keys())
    short = [SHORT_NAMES[n] for n in model_names]
    values = [predictions[n] for n in model_names]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    fig = go.Figure(go.Bar(
        x=short, y=values, marker_color=colors,
        text=[f'{v:.2f}' for v in values], textposition='outside',
    ))
    fig.update_layout(yaxis_title='Predicted Effort (Person-Months)', height=350)
    st.plotly_chart(fig, use_container_width=True)

    best_team = 3 if rec_effort > 6 else 2 if rec_effort > 3 else 1
    duration = rec_effort / best_team
    st.info(
        f"基于 {RECOMMENDED_MODEL} 模型，项目「{params['project_name']}」"
        f"预计工作量为 **{rec_effort:.2f} 人月**。"
        f"建议 {best_team} 人团队，工期约 {duration:.1f} 个月。"
    )
    return rec_effort


ensure_models()

add_compare = st.checkbox('添加对比项目（同时评估两个项目）')

if add_compare:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader('项目 A')
        params_a = render_input_form('a', '项目 A')
    with col_b:
        st.subheader('项目 B')
        params_b = render_input_form('b', '项目 B')
else:
    params_a = render_input_form('a', '新项目')
    params_b = None

if st.button('开始估算', type='primary'):
    models = st.session_state['trained_models']
    scaler = st.session_state['scaler']

    preds_a = predict_effort(models, scaler, params_a)

    if params_b:
        preds_b = predict_effort(models, scaler, params_b)
        col_a, col_b = st.columns(2)
        with col_a:
            rec_a = render_report(params_a, preds_a)
        with col_b:
            rec_b = render_report(params_b, preds_b)

        st.markdown('---')
        st.subheader('项目对比')
        compare_df = pd.DataFrame({
            '指标': ['推荐工作量 (人月)'],
            params_a['project_name']: [f"{rec_a:.2f}"],
            params_b['project_name']: [f"{rec_b:.2f}"],
            '差异': [f"{abs(rec_a - rec_b):.2f}"],
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.session_state['last_predictions'] = [(params_a, preds_a), (params_b, preds_b)]
    else:
        render_report(params_a, preds_a)
        st.session_state['last_predictions'] = [(params_a, preds_a)]

    st.markdown('---')

    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button('保存到数据库'):
            for params, preds in st.session_state['last_predictions']:
                save_estimation({
                    **{k: params[k] for k in [
                        'project_name', 'function_points', 'project_complexity',
                        'code_size_kloc', 'team_experience', 'tool_maturity',
                        'dev_mode', 'language_type',
                    ]},
                    'pred_ridge': preds['Linear Regression (Ridge)'],
                    'pred_dtree': preds['Decision Tree'],
                    'pred_svm': preds['SVM (RBF)'],
                    'pred_rf': preds['Random Forest'],
                    'recommended_effort': preds[RECOMMENDED_MODEL],
                })
            st.success('已保存到数据库！')

    with btn_cols[1]:
        for params, preds in st.session_state.get('last_predictions', []):
            pdf_bytes = generate_pdf(params, preds, RECOMMENDED_MODEL)
            st.download_button(
                f"下载 PDF — {params['project_name']}",
                data=pdf_bytes,
                file_name=f"estimation_{params['project_name']}.pdf",
                mime='application/pdf',
            )
