import streamlit as st
import plotly.express as px
import pandas as pd
from core.models import load_data, get_feature_importances

st.set_page_config(page_title='特征分析', layout='wide')
st.title('特征重要性分析')

_, X, y = load_data()
importances = get_feature_importances(X, y)

st.subheader('特征重要性排名')
imp_df = pd.DataFrame(importances, columns=['特征', '重要性得分'])
imp_df['重要性得分'] = imp_df['重要性得分'].round(4)
st.dataframe(imp_df, width="stretch", hide_index=True)

st.subheader('特征重要性可视化')
imp_df_sorted = imp_df.sort_values('重要性得分', ascending=True)
fig = px.bar(imp_df_sorted, x='重要性得分', y='特征', orientation='h',
             color='重要性得分', color_continuous_scale='RdYlGn',
             text=imp_df_sorted['重要性得分'].apply(lambda x: f'{x:.3f}'))
fig.update_layout(height=400, showlegend=False, yaxis_title='', xaxis_title='Feature Importance Score')
fig.update_traces(textposition='outside')
st.plotly_chart(fig, width="stretch")

top3 = imp_df.head(3)
st.subheader('分析结论')
st.info(
    f"特征重要性分析显示，影响软件项目工作量的前三大因素为："
    f"**{top3.iloc[0]['特征']}**（{top3.iloc[0]['重要性得分']:.3f}）、"
    f"**{top3.iloc[1]['特征']}**（{top3.iloc[1]['重要性得分']:.3f}）、"
    f"**{top3.iloc[2]['特征']}**（{top3.iloc[2]['重要性得分']:.3f}）。"
    f"其中 {top3.iloc[0]['特征']} 的影响最为显著。"
)
