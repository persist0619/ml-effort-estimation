import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='数据探索', layout='wide')
st.title('数据探索与预处理')

df = pd.read_csv('data/software_projects.csv')

st.subheader('基本统计信息')
st.dataframe(df.describe().round(2), use_container_width=True)

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
tabs = st.tabs(features)
for tab, feat in zip(tabs, features):
    with tab:
        fig = px.scatter(df, x=feat, y='actual_effort', opacity=0.6,
                         color_discrete_sequence=['steelblue'])
        fig.update_layout(xaxis_title=feat, yaxis_title='actual_effort')
        st.plotly_chart(fig, use_container_width=True)

st.subheader('特征相关性热力图')
corr = df.corr(numeric_only=True).round(2)
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1, aspect='auto')
fig.update_layout(width=700, height=600)
st.plotly_chart(fig, use_container_width=True)

st.subheader('箱线图（异常值检测）')
numeric_cols = ['function_points', 'project_complexity', 'code_size_kloc',
                'team_experience', 'tool_maturity', 'actual_effort']
selected = st.multiselect('选择要查看的特征', numeric_cols, default=numeric_cols)
if selected:
    fig = go.Figure()
    for col in selected:
        fig.add_trace(go.Box(y=df[col], name=col))
    fig.update_layout(yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)
