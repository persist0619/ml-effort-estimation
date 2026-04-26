import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.models import load_data, train_all_models, evaluate_models, MODEL_DEFS, SHORT_NAMES

st.set_page_config(page_title='模型训练', layout='wide')
st.title('模型训练与对比')

_, X, y = load_data()

st.subheader('模型训练')
if st.button('训练模型（用于预测）', type='primary'):
    with st.spinner('正在训练 4 个模型...'):
        trained, scaler = train_all_models(X, y)
        st.session_state['trained_models'] = trained
        st.session_state['scaler'] = scaler
    st.success('模型训练完成，可前往「工作量预测」页面进行预测。')

if 'trained_models' in st.session_state:
    st.info('模型已训练，可直接使用。')

st.markdown('---')

st.subheader('模型性能对比（10 次重复实验）')
if st.button('开始评估实验'):
    progress_bar = st.progress(0)
    status = st.empty()

    def on_progress(current, total):
        progress_bar.progress(current / total)
        status.text(f'第 {current}/{total} 次实验完成')

    with st.spinner('正在进行 10 次重复实验...'):
        summary = evaluate_models(X, y, n_repeats=10, progress_callback=on_progress)

    status.text('实验完成！')

    rows = []
    for name in MODEL_DEFS:
        avg = summary[name]['avg']
        rows.append({
            '模型': name,
            'MAE': f"{avg['MAE']:.2f}",
            'RMSE': f"{avg['RMSE']:.2f}",
            'MMRE': f"{avg['MMRE']:.3f}",
            'Pred(25)': f"{avg['Pred(25)']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    metrics = ['MAE', 'RMSE', 'MMRE', 'Pred(25)']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    model_names = list(MODEL_DEFS.keys())
    short = [SHORT_NAMES[n] for n in model_names]

    cols = st.columns(4)
    for idx, metric in enumerate(metrics):
        values = [summary[m]['avg'][metric] for m in model_names]
        stds = [summary[m]['std'][metric] for m in model_names]
        fig = go.Figure(go.Bar(
            x=short, y=values, marker_color=colors,
            error_y=dict(type='data', array=stds, visible=True),
            text=[f'{v:.3f}' for v in values], textposition='outside',
        ))
        fig.update_layout(title=metric, yaxis_title=metric, height=350)
        with cols[idx]:
            st.plotly_chart(fig, use_container_width=True)
