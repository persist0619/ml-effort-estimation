# 基于机器学习的软件项目工作量估算研究与应用

本科毕业论文配套演示系统，使用 Python 实现软件项目工作量估算的机器学习方法对比与分析。

## 项目结构

```
├── data/
│   └── software_projects.csv   # 模拟数据集（500条记录）
├── generate_data.py            # 数据生成脚本（运行一次即可）
├── 1_data_explore.py           # 数据探索与可视化
├── 2_model_train.py            # 模型训练与对比
├── 3_predict.py                # 工作量预测
├── 4_feature_analysis.py       # 特征重要性分析
├── requirements.txt            # Python 依赖
├── design.md                   # 系统设计文档
└── plan.md                     # 实施计划
```

## 环境准备

需要 Python 3.9+，推荐使用虚拟环境：

```bash
# 使用 uv（推荐）
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用 pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使用方法

按顺序运行以下脚本：

### 1. 数据探索

```bash
python3 1_data_explore.py
```

展示数据集的基本统计信息，弹出 4 张图表：
- 工作量分布直方图
- 各特征与工作量的散点图
- 特征相关性热力图
- 箱线图（异常值检测）

### 2. 模型训练与对比

```bash
python3 2_model_train.py
```

训练 4 个机器学习模型（线性回归、决策树、SVM、随机森林），通过网格搜索 + 5 折交叉验证调参，重复 10 次实验取平均值。输出：
- 控制台打印各模型的 MAE、RMSE、MMRE、Pred(25) 指标
- 弹出 4 指标对比柱状图

### 3. 工作量预测

```bash
python3 3_predict.py
```

使用训练好的模型对新项目进行工作量预测。修改脚本顶部的 `PROJECT_PARAMS` 字典即可输入不同的项目参数：

```python
PROJECT_PARAMS = {
    'function_points': 300,       # 功能点计数 (50-1500)
    'project_complexity': 3.0,    # 项目复杂度 (1.0-5.0)
    'code_size_kloc': 50,         # 代码规模/千行 (5-500)
    'team_experience': 3.5,       # 团队经验 (1.0-5.0)
    'tool_maturity': 3.0,         # 开发环境成熟度 (1.0-5.0)
    'dev_mode': 1,                # 开发模式 (0=有机型, 1=半分离型, 2=嵌入型)
    'language_type': 1,           # 语言类型 (0=低级, 1=高级, 2=超高级)
}
```

### 4. 特征重要性分析

```bash
python3 4_feature_analysis.py
```

基于随机森林模型分析各特征对工作量的影响程度，输出特征重要性排名和水平条形图。

## 数据集说明

`data/software_projects.csv` 包含 500 条模拟的软件项目数据，字段如下：

| 字段 | 说明 | 取值范围 |
|------|------|----------|
| function_points | 功能点计数 | 50-1500 |
| project_complexity | 项目复杂度 | 1.0-5.0 |
| code_size_kloc | 代码规模（千行） | 5-500 |
| team_experience | 团队经验水平 | 1.0-5.0 |
| tool_maturity | 开发环境成熟度 | 1.0-5.0 |
| dev_mode | 开发模式 | 0/1/2 |
| language_type | 语言类型 | 0/1/2 |
| actual_effort | 实际工作量（人月） | 1-100 |

如需重新生成数据：`python3 generate_data.py`

## 依赖

- scikit-learn >= 1.2.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
