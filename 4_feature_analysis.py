"""
特征重要性分析
功能：基于随机森林模型输出各特征重要性得分，绘制水平条形图
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('data/software_projects.csv')
feature_cols = ['function_points', 'project_complexity', 'code_size_kloc',
                'team_experience', 'tool_maturity', 'dev_mode', 'language_type']
feature_labels = [
    'Function Points',
    'Project Complexity',
    'Code Size (KLoC)',
    'Team Experience',
    'Tool Maturity',
    'Development Mode',
    'Language Type',
]

X = df[feature_cols].values
y = df['actual_effort'].values

# 训练随机森林
rf = RandomForestRegressor(n_estimators=200, max_depth=10, max_features='sqrt', random_state=42)
rf.fit(X, y)

# 特征重要性
importances = rf.feature_importances_
indices = np.argsort(importances)  # 升序，画图时从下到上

# 控制台打印
print("=" * 50)
print("特征重要性分析（随机森林）")
print("=" * 50)
sorted_idx = np.argsort(importances)[::-1]  # 降序
for i in sorted_idx:
    print(f"  {feature_labels[i]:<25} {importances[i]:.4f}")

# 绘制水平条形图
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_cols)))
bars = ax.barh(
    [feature_labels[i] for i in indices],
    importances[indices],
    color=colors, edgecolor='white', height=0.6
)

# 在条形右侧标注数值
for bar, val in zip(bars, importances[indices]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10)

ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.set_title('Random Forest Feature Importance Analysis', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(importances) * 1.2)
plt.tight_layout()
plt.show()

print("\n特征重要性分析完成。")
