"""
数据探索与预处理
功能：加载数据集，展示基本统计信息，绘制分布图、散点图、相关性热力图、箱线图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_csv('data/software_projects.csv')

print("=" * 60)
print("数据集基本信息")
print("=" * 60)
print(f"数据维度: {df.shape[0]} 行 x {df.shape[1]} 列")
print(f"\n缺失值统计:\n{df.isnull().sum()}")
print(f"\n描述性统计:\n{df.describe().round(2)}")

# 2. 工作量分布直方图
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['actual_effort'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('Actual Effort (Person-Months)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Software Project Effort', fontsize=14)
ax.axvline(df['actual_effort'].mean(), color='red', linestyle='--',
           label=f"Mean={df['actual_effort'].mean():.1f}")
ax.axvline(df['actual_effort'].median(), color='orange', linestyle='--',
           label=f"Median={df['actual_effort'].median():.1f}")
ax.legend()
plt.tight_layout()
plt.show()

# 3. 各特征与工作量的散点图
features = ['function_points', 'project_complexity', 'code_size_kloc',
            'team_experience', 'tool_maturity', 'dev_mode', 'language_type']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    axes[i].scatter(df[feat], df['actual_effort'], alpha=0.5, s=15, color='steelblue')
    axes[i].set_xlabel(feat, fontsize=9)
    axes[i].set_ylabel('actual_effort', fontsize=9)
    axes[i].set_title(f'{feat} vs Effort', fontsize=10)
axes[7].axis('off')
plt.suptitle('Feature vs Actual Effort Scatter Plots', fontsize=14)
plt.tight_layout()
plt.show()

# 4. 相关性热力图
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr().round(2)
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax, fmt='.2f')
ax.set_title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()

# 5. 箱线图检测异常值
numeric_cols = ['function_points', 'project_complexity', 'code_size_kloc',
                'team_experience', 'tool_maturity', 'actual_effort']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col], vert=True)
    axes[i].set_title(col, fontsize=10)
plt.suptitle('Box Plots for Outlier Detection', fontsize=14)
plt.tight_layout()
plt.show()

print("\n数据探索完成。")
