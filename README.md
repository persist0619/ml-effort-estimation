# 基于机器学习的软件项目工作量估算研究与应用

本科毕业论文配套演示系统。基于 Streamlit 构建的 Web 应用，实现软件项目工作量的机器学习估算、多模型对比、报告生成与历史数据管理。

## 快速启动

```bash
# 1. 创建虚拟环境并安装依赖
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

# 2. 启动 Web 应用
streamlit run app.py
```

浏览器打开 http://localhost:8501 即可使用。

## 系统功能

| 页面 | 功能 |
|------|------|
| 首页 | 系统概览、数据集统计 |
| 数据探索 | 分布直方图、散点图、相关性热力图、箱线图 |
| 模型训练 | 4 模型训练对比（10 次重复实验）、性能指标柱状图 |
| 工作量预测 | 输入项目参数 → 4 模型预测 → 工期换算 → PDF 报告下载 → 保存到数据库 |
| 特征分析 | 随机森林特征重要性排名与可视化 |
| 历史记录 | 历史估算数据查看、多项目对比、记录管理 |

支持同时评估两个项目并排对比。

## 项目结构

```
├── app.py                      # Streamlit 入口（首页）
├── pages/
│   ├── 1_数据探索.py            # 数据集统计与可视化
│   ├── 2_模型训练.py            # 模型训练与对比
│   ├── 3_工作量预测.py          # 预测 → 报告 → 存储（核心）
│   ├── 4_特征分析.py            # 特征重要性分析
│   └── 5_历史记录.py            # 历史数据与多项目对比
├── core/
│   ├── models.py               # 模型定义、训练、预测逻辑
│   ├── metrics.py              # MMRE、Pred(25) 等指标计算
│   ├── database.py             # SQLite 数据库操作
│   └── report.py               # PDF 报告生成
├── data/
│   └── software_projects.csv   # 模拟数据集（500 条）
├── generate_data.py            # 数据生成脚本
├── requirements.txt            # Python 依赖
└── estimation.db               # SQLite 数据库（自动创建）
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | Streamlit |
| 机器学习 | scikit-learn（Ridge、决策树、SVM、随机森林） |
| 交互式图表 | Plotly |
| 数据库 | SQLite |
| PDF 报告 | fpdf2 |

## 数据集说明

`data/software_projects.csv` 包含 500 条模拟数据，基于 COCOMO 模型特征分布构造。

| 字段 | 说明 | 取值范围 |
|------|------|----------|
| function_points | 功能点计数 | 50-1500 |
| project_complexity | 项目复杂度 | 1.0-5.0 |
| code_size_kloc | 代码规模（千行） | 5-500 |
| team_experience | 团队经验水平 | 1.0-5.0 |
| tool_maturity | 开发环境成熟度 | 1.0-5.0 |
| dev_mode | 开发模式 | 0=有机型 / 1=半分离型 / 2=嵌入型 |
| language_type | 语言类型 | 0=低级 / 1=高级 / 2=超高级 |
| actual_effort | 实际工作量（人月） | 3-35 |

如需重新生成：`python3 generate_data.py`

## 原始脚本

项目根目录下的 `1_data_explore.py`、`2_model_train.py`、`3_predict.py`、`4_feature_analysis.py` 为早期独立脚本版本，保留作为参考。Web 应用已将其逻辑迁移至 `core/` 模块和 `pages/` 页面中。
