# 软件项目工作量估算系统 — Web 应用升级设计

## 背景

当前系统是 4 个独立 Python 脚本，通过 matplotlib 弹窗展示图表，无数据存储，无统一界面。导师要求升级为完整的「输入-处理-输出」应用系统。

### 导师核心改进要求

1. 完善输出模块：生成包含工作量、工期、人力需求的完整估算报告
2. 新增数据库存储：历史估算数据永久留存
3. 优化交互逻辑：去除弹窗，明确各功能模块含义
4. 多项目同步评估：至少支持两个项目的同步评估与结果对比
5. 搭建完整应用框架：统一的数据输入、标准化处理、规范化输出

## 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| Web 框架 | Streamlit (multipage) | 纯 Python，开发效率最高，自带美观 UI |
| 数据库 | SQLite (内置 sqlite3) | 零配置，单文件，演示无风险 |
| 图表 | Plotly | 交互式图表，Streamlit 原生集成 |
| PDF 生成 | fpdf2 | 轻量，支持中文，纯 Python |
| ML | scikit-learn | 延续现有模型代码 |

## 文件结构

```
├── app.py                      # Streamlit 入口（系统首页）
├── pages/
│   ├── 1_数据探索.py            # 数据集统计与可视化
│   ├── 2_模型训练.py            # 4 模型训练对比
│   ├── 3_工作量预测.py          # 参数输入 → 预测 → 报告 → 存储
│   ├── 4_特征分析.py            # 特征重要性分析
│   └── 5_历史记录.py            # 历史数据查看与多项目对比
├── core/
│   ├── models.py               # 模型定义、训练、预测逻辑
│   ├── metrics.py              # MMRE、Pred(25) 等指标函数
│   ├── database.py             # SQLite 建表、CRUD 操作
│   └── report.py               # PDF 报告生成
├── .streamlit/
│   └── config.toml             # Streamlit 主题配置
├── data/
│   └── software_projects.csv
├── estimation.db               # SQLite 数据库（自动创建）
├── generate_data.py            # 数据生成脚本（保留）
└── requirements.txt
```

## 页面设计

### 首页 (app.py)

- 系统标题：「软件项目工作量估算系统」
- 系统简介：一段文字说明系统功能
- 数据集概况：样本数、特征数、工作量均值/中位数
- 快速入口：指向各功能页面的按钮或链接

### 1_数据探索

从现有 `1_data_explore.py` 迁移，改为 Plotly 交互式图表：

- 工作量分布直方图（带均值/中位数标记线）
- 各特征与工作量的散点图（2x4 网格）
- 特征相关性热力图
- 箱线图异常值检测
- 描述性统计表格

所有图表内嵌在页面中，不弹窗。

### 2_模型训练

从现有 `2_model_train.py` 迁移：

- 点击「开始训练」按钮触发 10 次重复实验
- 训练过程显示进度条
- 结果展示：4 模型 × 4 指标的对比表格
- 内嵌柱状图（MAE、RMSE、MMRE、Pred(25) 分组对比）
- 训练完成后模型缓存到 `st.session_state`，预测页面可直接使用

### 3_工作量预测（核心页面）

**输入区域：**
- 项目名称（文本框，必填）
- 7 个特征参数：
  - function_points: slider (50-1500)
  - project_complexity: slider (1.0-5.0, step 0.1)
  - code_size_kloc: slider (5-500)
  - team_experience: slider (1.0-5.0, step 0.1)
  - tool_maturity: slider (1.0-5.0, step 0.1)
  - dev_mode: selectbox (有机型/半分离型/嵌入型)
  - language_type: selectbox (低级/高级/超高级)
- 「添加对比项目」按钮：展开第二组相同的参数表单
- 「开始估算」按钮

**输出区域（报告）：**

对每个项目生成以下报告内容：

1. **项目基本信息表** — 7 个输入参数的汇总
2. **四模型预测结果对比表** — 模型名、预测工作量（人月），最优模型高亮
3. **工期与人力换算表** — 基于推荐模型预测值：

   | 团队规模 | 预计工期 |
   |---------|---------|
   | 1 人 | X 个月 |
   | 2 人 | X 个月 |
   | 3 人 | X 个月 |
   | 5 人 | X 个月 |

4. **预测结果柱状图** — 4 模型预测值对比
5. **结论与建议** — 自动生成文字：「基于 XX 模型，本项目预计工作量为 XX 人月。建议 X 人团队，工期约 X 个月。」

**多项目对比：** 如果输入了两个项目，两组报告并排展示（Streamlit columns），底部增加对比汇总表。

**操作按钮：**
- 「保存到数据库」— 将估算结果写入 SQLite
- 「下载 PDF 报告」— 生成并下载 PDF 文件

### 4_特征分析

从现有 `4_feature_analysis.py` 迁移：

- 特征重要性水平条形图（Plotly）
- 特征重要性得分表格
- 简要分析结论文字

### 5_历史记录

- 全部历史估算记录表格（从 SQLite 读取）
- 支持按项目名、时间范围筛选
- 勾选多条记录 → 生成多项目对比柱状图（各模型预测值并排对比）
- 删除记录功能

## 数据库设计

单表 `estimations`：

```sql
CREATE TABLE IF NOT EXISTS estimations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    function_points REAL,
    project_complexity REAL,
    code_size_kloc REAL,
    team_experience REAL,
    tool_maturity REAL,
    dev_mode INTEGER,
    language_type INTEGER,
    pred_ridge REAL,
    pred_dtree REAL,
    pred_svm REAL,
    pred_rf REAL,
    recommended_effort REAL
);
```

## core 模块设计

### core/models.py

- `train_all_models(X, y)` — 训练 4 个模型（含网格搜索），返回训练好的模型字典
- `predict_effort(models, scaler, params)` — 输入参数字典，返回 4 个模型的预测值
- `evaluate_models(X, y, n_repeats=10)` — 10 次重复实验，返回汇总指标

### core/metrics.py

- `calc_mmre(y_true, y_pred)` — 计算 MMRE
- `calc_pred25(y_true, y_pred)` — 计算 Pred(25)
- `evaluate_model(y_true, y_pred)` — 返回 MAE/RMSE/MMRE/Pred(25) 字典

### core/database.py

- `init_db()` — 建表（如不存在）
- `save_estimation(data: dict)` — 插入一条估算记录
- `get_all_estimations()` — 查询全部记录，返回 DataFrame
- `delete_estimation(id: int)` — 删除指定记录
- `get_estimations_by_ids(ids: list)` — 按 ID 列表查询

### core/report.py

- `generate_pdf(project_data: dict, predictions: dict, chart_path: str)` — 生成 PDF 报告，返回文件字节流

PDF 内容：项目信息表 + 预测结果表 + 工期换算表 + 图表 + 结论文字。使用 fpdf2 + 中文字体（SimHei 或系统自带字体）。

## 模型训练策略

- 系统启动后，用户访问「模型训练」页面点击训练，或「工作量预测」页面自动检测未训练则提示
- 训练结果（模型对象 + scaler）缓存到 `st.session_state`
- 预测页面直接从 session_state 取模型进行预测

## 错误处理

- 输入参数：Streamlit slider/selectbox 自带范围约束，无需额外校验
- 模型未训练：预测页面显示提示并引导用户前往训练页面
- 数据库异常：try/except 包裹，`st.error()` 展示错误信息

## 现有代码复用

现有 4 个脚本的核心逻辑提取到 `core/` 模块复用。原始脚本保留在项目根目录作为参考，但不再作为系统入口。

## 新增依赖

```
streamlit>=1.30.0
fpdf2>=2.7.0
plotly>=5.18.0
```

追加到现有 requirements.txt 中。
