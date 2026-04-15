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
        'params': {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.1], 'epsilon': [0.1, 0.5]},
        'need_scale': True,
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 15, None],
                   'max_features': ['sqrt', 'log2', 0.5], 'min_samples_leaf': [1, 2, 5]},
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
