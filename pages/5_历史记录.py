import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.database import get_all_estimations, delete_estimation, get_estimations_by_ids

st.set_page_config(page_title='历史记录', layout='wide')
st.title('历史估算记录')

df = get_all_estimations()

if df.empty:
    st.info('暂无历史记录。请先在「工作量预测」页面进行估算并保存。')
    st.stop()

st.subheader('筛选')
col1, col2 = st.columns(2)
with col1:
    name_filter = st.text_input('按项目名称筛选')
with col2:
    if 'created_at' in df.columns and not df['created_at'].isna().all():
        df['created_at'] = pd.to_datetime(df['created_at'])

filtered = df.copy()
if name_filter:
    filtered = filtered[filtered['project_name'].str.contains(name_filter, case=False, na=False)]

st.subheader(f'记录列表（共 {len(filtered)} 条）')
display_cols = ['id', 'project_name', 'created_at', 'function_points',
                'project_complexity', 'pred_svm', 'pred_rf', 'recommended_effort']
available_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

st.subheader('多项目对比')
if len(filtered) >= 2:
    selected_ids = st.multiselect(
        '选择要对比的记录（按 ID）',
        filtered['id'].tolist(),
        default=filtered['id'].tolist()[:2],
    )
    if len(selected_ids) >= 2:
        compare_df = get_estimations_by_ids(selected_ids)
        model_cols = {'pred_ridge': 'LinReg', 'pred_dtree': 'DTree',
                      'pred_svm': 'SVM', 'pred_rf': 'RF'}
        fig = go.Figure()
        for _, row in compare_df.iterrows():
            values = [row[col] for col in model_cols.keys()]
            fig.add_trace(go.Bar(
                name=row['project_name'],
                x=list(model_cols.values()),
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='outside',
            ))
        fig.update_layout(
            barmode='group', yaxis_title='Predicted Effort (Person-Months)',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.caption('至少需要 2 条记录才能进行对比。')

st.subheader('管理')
del_id = st.number_input('输入要删除的记录 ID', min_value=0, step=1)
if st.button('删除记录', type='secondary'):
    if del_id > 0:
        delete_estimation(int(del_id))
        st.success(f'已删除记录 ID={del_id}')
        st.rerun()
    else:
        st.warning('请输入有效的记录 ID')
