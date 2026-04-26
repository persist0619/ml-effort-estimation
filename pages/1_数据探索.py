import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='数据探索', layout='wide')
st.title('数据探索与预处理')

COL_LABELS = {
    'function_points': '功能点计数',
    'project_complexity': '项目复杂度',
    'code_size_kloc': '代码规模(千行)',
    'team_experience': '团队经验',
    'tool_maturity': '工具成熟度',
    'dev_mode': '开发模式',
    'language_type': '语言类型',
    'actual_effort': '实际工作量(人月)',
}

df = pd.read_csv('data/software_projects.csv')
df_display = df.rename(columns=COL_LABELS)

st.subheader('基本统计信息')
st.dataframe(df_display.describe().round(2), use_container_width=True)

missing = df.isnull().sum()
if missing.sum() == 0:
    st.success('数据集无缺失值')
else:
    st.warning(f'缺失值: {missing[missing > 0].to_dict()}')

st.subheader('工作量分布')
fig = px.histogram(df, x='actual_effort', nbins=30, color_discrete_sequence=['steelblue'])
fig.add_vline(x=df['actual_effort'].mean(), line_dash='dash', line_color='red',
              annotation_text=f"Mean={df['actual_effort'].mean():.1f}")
fig.add_vline(x=df['actual_effort'].median(), line_dash='dash', line_color='orange',
              annotation_text=f"Median={df['actual_effort'].median():.1f}")
fig.update_layout(xaxis_title='Actual Effort (Person-Months)', yaxis_title='Frequency')
st.plotly_chart(fig, use_container_width=True)

st.subheader('各特征与工作量的关系')
features = ['function_points', 'project_complexity', 'code_size_kloc',
            'team_experience', 'tool_maturity', 'dev_mode', 'language_type']
tab_labels = [f"{COL_LABELS[f]}" for f in features]
tabs = st.tabs(tab_labels)
for tab, feat in zip(tabs, features):
    with tab:
        fig = px.scatter(df, x=feat, y='actual_effort', opacity=0.6,
                         color_discrete_sequence=['steelblue'])
        fig.update_layout(xaxis_title=COL_LABELS[feat], yaxis_title=COL_LABELS['actual_effort'])
        st.plotly_chart(fig, use_container_width=True)

st.subheader('特征相关性热力图')
corr = df_display.corr(numeric_only=True).round(2)
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1, aspect='auto')
fig.update_layout(width=700, height=600)
st.plotly_chart(fig, use_container_width=True)

st.subheader('箱线图（异常值检测）')
numeric_cols = ['function_points', 'project_complexity', 'code_size_kloc',
                'team_experience', 'tool_maturity', 'actual_effort']
box_labels = [COL_LABELS[c] for c in numeric_cols]
selected = st.multiselect('选择要查看的特征', box_labels, default=box_labels)
if selected:
    label_to_col = {v: k for k, v in COL_LABELS.items()}
    fig = go.Figure()
    for label in selected:
        col = label_to_col[label]
        fig.add_trace(go.Box(y=df[col], name=label))
    fig.update_layout(yaxis_title='数值')
    st.plotly_chart(fig, use_container_width=True)
