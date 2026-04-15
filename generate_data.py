"""
数据生成脚本 - 生成模拟的软件项目工作量数据集
运行一次即可，生成 data/software_projects.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

# 生成特征
function_points = np.random.uniform(50, 1500, n)
project_complexity = np.random.uniform(1.0, 5.0, n)
code_size_kloc = np.random.uniform(5, 500, n)
team_experience = np.random.uniform(1.0, 5.0, n)
tool_maturity = np.random.uniform(1.0, 5.0, n)
dev_mode = np.random.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
language_type = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])

# 非线性工作量公式：三种开发模式对应完全不同的工作量计算方式
# 树模型可以自然地对 dev_mode 做分裂，线性模型/SVM 即使独热编码也难以拟合

# 有机型：FP 主导，复杂度影响小
effort_organic = 0.30 * function_points**0.55 * (0.8 + 0.1 * project_complexity)
# 半分离型：FP 和代码规模共同作用，复杂度放大效应强
effort_semi = 0.15 * function_points**0.5 * project_complexity**0.8 + 0.05 * code_size_kloc**0.7
# 嵌入型：FP 的高次方效应，小项目和大项目差距悬殊
effort_embedded = 0.02 * function_points**0.7 * project_complexity**0.5 + 7.0

base = np.where(dev_mode == 0, effort_organic,
       np.where(dev_mode == 1, effort_semi, effort_embedded))

# 团队经验效应因模式而异
team_penalty = np.where(
    dev_mode == 2,
    5.0 * np.maximum(3.5 - team_experience, 0),   # 嵌入型：经验不足惩罚极大
    2.0 * np.maximum(3.0 - team_experience, 0)     # 其他：经验不足惩罚温和
)

tool_effect = 1.2 * np.maximum(3.0 - tool_maturity, 0)

# 语言类型：非单调效应
lang_effect = np.where(language_type == 0, 3.0,
              np.where(language_type == 1, 0.0, 1.5))

effort = base + team_penalty + tool_effect + lang_effect

# 对数正态乘法噪声
noise = np.random.lognormal(0, 0.12, n)
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
