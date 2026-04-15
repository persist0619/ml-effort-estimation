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

# 非线性工作量公式：分类特征与连续特征深度交互，树模型占优
# function_points 的效应系数取决于 dev_mode（分类×连续交互）
fp_coeff = np.where(dev_mode == 0, 0.22, np.where(dev_mode == 1, 0.33, 0.28))
base = fp_coeff * function_points**0.5 * project_complexity**0.55

# 代码规模
code_contrib = 0.012 * code_size_kloc**0.75

# 团队和工具效率（非线性除法关系）
team_factor = 1.0 / (0.5 + 0.15 * team_experience)
tool_factor = 1.0 / (0.6 + 0.12 * tool_maturity)

# 语言类型：非单调效应（1 最低，0 和 2 较高），线性模型无法拟合
lang_factor = np.where(language_type == 0, 1.15,
              np.where(language_type == 1, 0.88, 1.08))

effort = (base + code_contrib) * team_factor * tool_factor * lang_factor

# 对数正态乘法噪声
noise = np.random.lognormal(0, 0.10, n)
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
