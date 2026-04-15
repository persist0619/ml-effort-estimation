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
