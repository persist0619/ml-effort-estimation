# 基于机器学习的软件项目工作量估算 — 演示系统设计

## 概述

为本科毕业论文《基于机器学习的软件项目工作量估算研究与应用》配套的纯 Python 演示脚本，用于答辩现场运行展示。

## 文件结构

```
├── data/
│   └── software_projects.csv
├── 1_data_explore.py
├── 2_model_train.py
├── 3_predict.py
├── 4_feature_analysis.py
└── requirements.txt
```

## 数据集：`data/software_projects.csv`

约 200 条模拟数据，基于论文描述的 Promise Repository（COCOMO/NASA/Desharnais）特征分布构造。

### 字段定义

| 字段 | 类型 | 说明 | 取值范围 |
|------|------|------|----------|
| function_points | float | 功能点计数 | 50-1500 |
| project_complexity | float | 项目复杂度（1-5 等级） | 1.0-5.0 |
| code_size_kloc | float | 代码规模（千行） | 5-500 |
| team_experience | float | 团队经验水平（1-5 等级） | 1.0-5.0 |
| tool_maturity | float | 开发环境成熟度（1-5 等级） | 1.0-5.0 |
| dev_mode | int | 开发模式（0=有机型, 1=半分离型, 2=嵌入型） | 0/1/2 |
| language_type | int | 语言类型（0=低级, 1=高级, 2=超高级） | 0/1/2 |
| actual_effort | float | 实际工作量（人月） | 1-100 |

### 数据生成逻辑

- actual_effort 与各特征之间存在非线性关系，function_points 影响最大
- 加入适度噪声模拟真实数据波动
- 数据分布呈右偏（大多数项目工作量较小，少数大型项目工作量很高）
- 最终模型结果应接近论文结论：随机森林 MAE≈2.18, Pred(25)≈0.708

## 脚本设计

### 1_data_explore.py — 数据探索与预处理

**输入**：`data/software_projects.csv`

**功能**：
1. 加载数据，打印基本统计信息（shape、describe、缺失值）
2. 绘制目标变量 actual_effort 的分布直方图
3. 绘制各特征与 actual_effort 的散点图（2x4 子图）
4. 绘制特征间相关性热力图
5. 绘制箱线图检测异常值

**输出**：matplotlib 图表弹窗 + 控制台统计信息

### 2_model_train.py — 模型训练与对比

**输入**：`data/software_projects.csv`

**功能**：
1. 数据预处理：分类特征独热编码，连续特征标准化（SVM 用 Z-score，树模型不标准化）
2. 70/30 划分训练集/测试集（分层抽样）
3. 训练 4 个模型：
   - 多元线性回归（带 Ridge 正则化，alpha 网格搜索）
   - 决策树（max_depth、min_samples_split 网格搜索）
   - SVM-RBF（C、gamma、epsilon 网格搜索）
   - 随机森林（n_estimators、max_depth、max_features 网格搜索）
4. 每个模型用 5 折交叉验证选参
5. 在测试集上计算 MAE、RMSE、MMRE、Pred(25)
6. 重复 10 次实验取平均值和标准差
7. 绘制 4 指标对比柱状图
8. 控制台打印结果表格

**输出**：对比柱状图 + 控制台结果表

### 3_predict.py — 工作量预测

**输入**：用户在代码顶部修改的参数字典

**功能**：
1. 加载数据，训练 4 个模型（复用 2 的逻辑）
2. 根据用户输入的项目参数，4 个模型分别预测
3. 打印各模型预测结果 + 推荐使用随机森林结果

**输出**：控制台打印预测结果

### 4_feature_analysis.py — 特征重要性分析

**输入**：`data/software_projects.csv`

**功能**：
1. 训练随机森林模型
2. 提取 feature_importances_
3. 绘制水平条形图，按重要性降序排列
4. 控制台打印各特征重要性得分

**输出**：条形图 + 控制台打印

## 依赖

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## 不做的事

- 不做 Web 界面
- 不做数据库
- 不做用户登录
- 不做文件上传
- 不做模型持久化（pkl）
- 不做复杂的命令行交互
