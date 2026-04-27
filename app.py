import os
import streamlit as st
import pandas as pd
from core.database import init_db

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

init_db()

st.set_page_config(
    page_title='软件项目工作量估算系统',
    page_icon='📊',
    layout='wide',
)

st.title('软件项目工作量估算系统')
st.markdown('基于机器学习的软件项目工作量估算研究与应用 — 演示系统')

st.markdown('---')

st.subheader('系统功能')
col1, col2 = st.columns(2)
with col1:
    st.markdown('''
    - **数据探索** — 数据集统计分析与可视化
    - **模型训练** — 4 种机器学习模型训练与对比
    - **工作量预测** — 输入项目参数，生成估算报告
    ''')
with col2:
    st.markdown('''
    - **特征分析** — 特征重要性排名与分析
    - **历史记录** — 历史估算数据管理与多项目对比
    ''')

st.markdown('---')

st.subheader('数据集概况')
df = pd.read_csv(os.path.join(_PROJECT_ROOT, 'data', 'software_projects.csv'))
c1, c2, c3, c4 = st.columns(4)
c1.metric('样本数量', f'{len(df)} 条')
c2.metric('特征数量', '7 个')
c3.metric('工作量均值', f'{df["actual_effort"].mean():.2f} 人月')
c4.metric('工作量中位数', f'{df["actual_effort"].median():.2f} 人月')
