# 软件项目工作量估算演示系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 4 个 Python 演示脚本 + 1 个模拟数据集，完整展示论文中的机器学习工作量估算研究流程。

**Architecture:** 纯脚本架构，无框架依赖。一个数据生成脚本生成 CSV，4 个独立脚本分别完成数据探索、模型训练对比、预测、特征分析。脚本间无代码共享，各自独立读取 CSV。

**Tech Stack:** Python 3.9+, scikit-learn, pandas, numpy, matplotlib, seaborn

---

## File Map

| 文件 | 职责 |
|------|------|
| `requirements.txt` | 依赖声明 |
| `generate_data.py` | 一次性数据生成脚本，运行后生成 CSV（不参与答辩演示） |
| `data/software_projects.csv` | 生成的模拟数据集 |
| `1_data_explore.py` | 数据探索与可视化 |
| `2_model_train.py` | 4 模型训练 + 对比 |
| `3_predict.py` | 输入参数预测工作量 |
| `4_feature_analysis.py` | 特征重要性分析 |

---

### Task 1: 项目初始化 + 数据集生成

**Files:**
- Create: `requirements.txt`
- Create: `data/` 目录
- Create: `generate_data.py`
- Create: `data/software_projects.csv`（由脚本生成）

- [ ] **Step 1: 创建 requirements.txt**

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

- [ ] **Step 2: 创建 data 目录**

```bash
mkdir -p data
```

- [ ] **Step 3: 创建 generate_data.py**

此脚本用非线性公式 + 噪声生成约 200 条模拟数据，使得:
- function_points 对 actual_effort 影响最大（重要性 ~0.31）
- 数据呈右偏分布
- 随机森林训练后 MAE 接近 2-3 范围

```python
"""
数据生成脚本 - 生成模拟的软件项目工作量数据集
运行一次即可，生成 data/software_projects.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 200

# 生成特征
function_points = np.random.uniform(50, 1500, n)
project_complexity = np.random.uniform(1.0, 5.0, n)
code_size_kloc = np.random.uniform(5, 500, n)
team_experience = np.random.uniform(1.0, 5.0, n)
tool_maturity = np.random.uniform(1.0, 5.0, n)
dev_mode = np.random.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
language_type = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])

# 非线性工作量生成公式
# function_points 影响最大，其次是 complexity 和 code_size
effort = (
    0.08 * np.sqrt(function_points) * project_complexity
    + 0.015 * code_size_kloc ** 0.9
    + 1.5 * (5.0 - team_experience)
    + 0.8 * (5.0 - tool_maturity)
    + np.where(dev_mode == 2, 3.0, np.where(dev_mode == 1, 1.5, 0.0))
    + np.where(language_type == 0, 2.0, np.where(language_type == 1, 0.5, 0.0))
)

# 加入噪声（乘性 + 加性）
noise = np.random.normal(1.0, 0.15, n) + np.random.normal(0, 0.5, n)
actual_effort = np.maximum(effort * noise, 1.0)
actual_effort = np.round(actual_effort, 2)

# 构建 DataFrame
df = pd.DataFrame({
    'function_points': np.round(function_points, 1),
    'project_complexity': np.round(project_complexity, 2),
    'code_size_kloc': np.round(code_size_kloc, 1),
    'team_experience': np.round(team_experience, 2),
    'tool_maturity': np.round(tool_maturity, 2),
    'dev_mode': dev_mode,
    'language_type': language_type,
    'actual_effort': actual_effort,
})

df.to_csv('data/software_projects.csv', index=False)
print(f"数据集已生成: {df.shape[0]} 条记录")
print(f"\n工作量统计:")
print(f"  均值: {df['actual_effort'].mean():.2f} 人月")
print(f"  中位数: {df['actual_effort'].median():.2f} 人月")
print(f"  最小值: {df['actual_effort'].min():.2f} 人月")
print(f"  最大值: {df['actual_effort'].max():.2f} 人月")
print(f"\n前5条数据:")
print(df.head())
```

- [ ] **Step 4: 运行数据生成脚本**

```bash
python3 generate_data.py
```

预期输出：控制台打印数据统计信息，`data/software_projects.csv` 文件生成。

- [ ] **Step 5: 验证 CSV 数据**

```bash
head -5 data/software_projects.csv
wc -l data/software_projects.csv
```

预期：CSV 有表头 + 200 行数据，共 201 行。

---

### Task 2: 数据探索脚本

**Files:**
- Create: `1_data_explore.py`

- [ ] **Step 1: 创建 1_data_explore.py**

```python
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
ax.axvline(df['actual_effort'].mean(), color='red', linestyle='--', label=f"Mean={df['actual_effort'].mean():.1f}")
ax.axvline(df['actual_effort'].median(), color='orange', linestyle='--', label=f"Median={df['actual_effort'].median():.1f}")
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
```

- [ ] **Step 2: 运行验证**

```bash
python3 1_data_explore.py
```

预期：控制台打印统计信息，弹出 4 张图表（分布直方图、散点图、热力图、箱线图）。

---

### Task 3: 模型训练与对比脚本

**Files:**
- Create: `2_model_train.py`

- [ ] **Step 1: 创建 2_model_train.py**

```python
"""
模型训练与对比
功能：训练线性回归、决策树、SVM、随机森林 4 个模型
     网格搜索 + 5折交叉验证调参
     计算 MAE、RMSE、MMRE、Pred(25) 并绘制对比柱状图
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False


def calc_mmre(y_true, y_pred):
    """计算 MMRE（平均相对误差）"""
    mre = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-6)
    return np.mean(mre)


def calc_pred25(y_true, y_pred):
    """计算 Pred(25)：相对误差 <= 25% 的样本比例"""
    mre = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-6)
    return np.mean(mre <= 0.25)


def evaluate_model(y_true, y_pred):
    """计算 4 个评价指标"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MMRE': calc_mmre(y_true, y_pred),
        'Pred(25)': calc_pred25(y_true, y_pred),
    }


# 加载数据
df = pd.read_csv('data/software_projects.csv')
feature_cols = ['function_points', 'project_complexity', 'code_size_kloc',
                'team_experience', 'tool_maturity', 'dev_mode', 'language_type']
X = df[feature_cols].values
y = df['actual_effort'].values

# 模型定义 + 参数搜索空间
models = {
    'Linear Regression (Ridge)': {
        'model': Ridge(),
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'need_scale': True,
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor(random_state=42),
        'params': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [5, 10, 15]},
        'need_scale': False,
    },
    'SVM (RBF)': {
        'model': SVR(kernel='rbf'),
        'params': {'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1.0], 'epsilon': [0.1, 0.5]},
        'need_scale': True,
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, 15], 'max_features': ['sqrt', 'log2']},
        'need_scale': False,
    },
}

# 重复 10 次实验
n_repeats = 10
all_results = {name: [] for name in models}

print("=" * 60)
print("模型训练与对比（10次重复实验）")
print("=" * 60)

for i in range(n_repeats):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, cfg in models.items():
        X_tr = X_train_scaled if cfg['need_scale'] else X_train
        X_te = X_test_scaled if cfg['need_scale'] else X_test

        grid = GridSearchCV(
            cfg['model'], cfg['params'], cv=5,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        grid.fit(X_tr, y_train)
        y_pred = grid.predict(X_te)
        metrics = evaluate_model(y_test, y_pred)
        all_results[name].append(metrics)

    print(f"  第 {i+1}/10 次实验完成")

# 汇总结果
print("\n" + "=" * 60)
print(f"{'模型':<25} {'MAE':>8} {'RMSE':>8} {'MMRE':>8} {'Pred(25)':>10}")
print("-" * 60)

summary = {}
for name, results_list in all_results.items():
    avg = {}
    std = {}
    for metric in ['MAE', 'RMSE', 'MMRE', 'Pred(25)']:
        values = [r[metric] for r in results_list]
        avg[metric] = np.mean(values)
        std[metric] = np.std(values)
    summary[name] = {'avg': avg, 'std': std}
    print(f"{name:<25} {avg['MAE']:>7.2f}  {avg['RMSE']:>7.2f}  {avg['MMRE']:>7.3f}  {avg['Pred(25)']:>9.3f}")

print("-" * 60)
print("(以上为 10 次实验平均值)")

# 绘制对比柱状图
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
metric_names = ['MAE', 'RMSE', 'MMRE', 'Pred(25)']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
model_names = list(summary.keys())
short_names = ['LinReg', 'DTree', 'SVM', 'RF']

for idx, metric in enumerate(metric_names):
    values = [summary[m]['avg'][metric] for m in model_names]
    stds = [summary[m]['std'][metric] for m in model_names]
    bars = axes[idx].bar(short_names, values, color=colors, yerr=stds, capsize=4, alpha=0.85)
    axes[idx].set_title(metric, fontsize=13, fontweight='bold')
    axes[idx].set_ylabel(metric, fontsize=10)
    for bar, val in zip(bars, values):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Model Performance Comparison (10 Repeated Experiments)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n模型训练与对比完成。")
```

- [ ] **Step 2: 运行验证**

```bash
python3 2_model_train.py
```

预期：控制台打印 10 次实验进度和结果表格，弹出 4 指标对比柱状图。随机森林各指标应优于其他模型。

---

### Task 4: 工作量预测脚本

**Files:**
- Create: `3_predict.py`

- [ ] **Step 1: 创建 3_predict.py**

```python
"""
工作量预测
功能：根据输入的项目参数，用 4 个模型分别预测工作量
使用方法：修改下方 PROJECT_PARAMS 字典中的参数值，然后运行脚本
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# ========================================
# 在这里修改项目参数
# ========================================
PROJECT_PARAMS = {
    'function_points': 300,       # 功能点计数 (50-1500)
    'project_complexity': 3.0,    # 项目复杂度 (1.0-5.0)
    'code_size_kloc': 50,         # 代码规模/千行 (5-500)
    'team_experience': 3.5,       # 团队经验 (1.0-5.0)
    'tool_maturity': 3.0,         # 开发环境成熟度 (1.0-5.0)
    'dev_mode': 1,                # 开发模式 (0=有机型, 1=半分离型, 2=嵌入型)
    'language_type': 1,           # 语言类型 (0=低级, 1=高级, 2=超高级)
}
# ========================================

# 加载数据并训练模型
df = pd.read_csv('data/software_projects.csv')
feature_cols = list(PROJECT_PARAMS.keys())
X = df[feature_cols].values
y = df['actual_effort'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 定义模型
models = {
    'Linear Regression (Ridge)': {'model': Ridge(alpha=1.0), 'need_scale': True},
    'Decision Tree': {'model': DecisionTreeRegressor(max_depth=7, min_samples_split=10, random_state=42), 'need_scale': False},
    'SVM (RBF)': {'model': SVR(kernel='rbf', C=10, gamma=0.1, epsilon=0.1), 'need_scale': True},
    'Random Forest': {'model': RandomForestRegressor(n_estimators=200, max_depth=10, max_features='sqrt', random_state=42), 'need_scale': False},
}

# 训练 + 预测
input_array = np.array([[PROJECT_PARAMS[f] for f in feature_cols]])
input_scaled = scaler.transform(input_array)

print("=" * 50)
print("软件项目工作量预测")
print("=" * 50)
print("\n输入参数:")
for k, v in PROJECT_PARAMS.items():
    print(f"  {k}: {v}")

print("\n" + "-" * 50)
print(f"{'模型':<28} {'预测工作量 (人月)':>15}")
print("-" * 50)

predictions = {}
for name, cfg in models.items():
    X_tr = X_train_scaled if cfg['need_scale'] else X_train
    inp = input_scaled if cfg['need_scale'] else input_array

    cfg['model'].fit(X_tr, y_train)
    pred = cfg['model'].predict(inp)[0]
    predictions[name] = pred
    print(f"{name:<28} {pred:>12.2f}")

print("-" * 50)
rf_pred = predictions['Random Forest']
print(f"\n>>> 推荐结果（随机森林）: {rf_pred:.2f} 人月")
print(f"    即约 {rf_pred:.1f} 个人工作 1 个月，或 1 个人工作 {rf_pred:.1f} 个月")
```

- [ ] **Step 2: 运行验证**

```bash
python3 3_predict.py
```

预期：控制台打印输入参数和 4 个模型的预测结果，推荐随机森林结果。

---

### Task 5: 特征重要性分析脚本

**Files:**
- Create: `4_feature_analysis.py`

- [ ] **Step 1: 创建 4_feature_analysis.py**

```python
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
```

- [ ] **Step 2: 运行验证**

```bash
python3 4_feature_analysis.py
```

预期：控制台打印各特征重要性得分（function_points 应排第一），弹出水平条形图。

---

### Task 6: 最终验证

- [ ] **Step 1: 按顺序运行全部脚本确认无报错**

```bash
python3 generate_data.py
python3 1_data_explore.py
python3 2_model_train.py
python3 3_predict.py
python3 4_feature_analysis.py
```

- [ ] **Step 2: 检查数据生成公式是否需要微调**

如果随机森林 MAE 与论文结论（~2.18）偏差过大，回到 `generate_data.py` 调整噪声参数重新生成数据。

- [ ] **Step 3: 确认所有图表显示正常**

检查中文字体渲染、图表标题、坐标轴标签是否清晰可读。
