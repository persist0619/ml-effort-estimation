# Web 应用升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将纯脚本演示系统升级为 Streamlit 多页面 Web 应用，包含数据库存储、PDF 报告生成、多项目对比功能。

**Architecture:** Streamlit multipage app，core/ 模块封装业务逻辑（模型训练、指标计算、数据库、PDF），pages/ 目录下 5 个页面文件各负责一个功能模块。SQLite 单文件数据库存储历史估算记录。

**Tech Stack:** Python 3.9+, Streamlit, scikit-learn, Plotly, SQLite3, fpdf2

---

## File Map

| 文件 | 职责 | 来源 |
|------|------|------|
| `requirements.txt` | 依赖声明 | 修改现有 |
| `.streamlit/config.toml` | Streamlit 主题配置 | 新建 |
| `core/__init__.py` | 包初始化 | 新建 |
| `core/metrics.py` | MMRE、Pred(25) 等指标计算 | 从 `2_model_train.py` 提取 |
| `core/models.py` | 模型定义、训练、预测 | 从 `2_model_train.py` + `3_predict.py` 提取 |
| `core/database.py` | SQLite 建表、CRUD | 新建 |
| `core/report.py` | PDF 报告生成 | 新建 |
| `app.py` | Streamlit 首页 | 新建 |
| `pages/1_数据探索.py` | 数据可视化 | 从 `1_data_explore.py` 迁移为 Plotly |
| `pages/2_模型训练.py` | 模型训练与对比 | 从 `2_model_train.py` 迁移 |
| `pages/3_工作量预测.py` | 输入→预测→报告→存储 | 新建（核心页面） |
| `pages/4_特征分析.py` | 特征重要性分析 | 从 `4_feature_analysis.py` 迁移 |
| `pages/5_历史记录.py` | 历史数据查看与多项目对比 | 新建 |

---

### Task 1: 项目基础设施

**Files:**
- Modify: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `core/__init__.py`

- [ ] **Step 1: 更新 requirements.txt**

```
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.30.0
plotly>=5.18.0
fpdf2>=2.7.0
```

- [ ] **Step 2: 创建 .streamlit/config.toml**

```bash
mkdir -p .streamlit
```

文件内容：

```toml
[server]
headless = true

[theme]
primaryColor = "#4C72B0"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

- [ ] **Step 3: 创建 core/__init__.py**

```python
```

空文件，使 core 成为 Python 包。

- [ ] **Step 4: 安装新依赖**

```bash
source .venv/bin/activate && uv pip install -r requirements.txt
```

预期：streamlit、plotly、fpdf2 安装成功。

- [ ] **Step 5: 验证 streamlit 可运行**

```bash
source .venv/bin/activate && python -c "import streamlit; print(streamlit.__version__)"
```

预期：打印版本号，无报错。

- [ ] **Step 6: 提交**

```bash
git add requirements.txt .streamlit/config.toml core/__init__.py
git commit -m "chore: add streamlit/plotly/fpdf2 deps and project structure"
```

---

### Task 2: core/metrics.py — 指标计算模块

**Files:**
- Create: `core/metrics.py`

- [ ] **Step 1: 创建 core/metrics.py**

从现有 `2_model_train.py:23-42` 提取三个函数：

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calc_mmre(y_true, y_pred):
    mre = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-6)
    return np.mean(mre)


def calc_pred25(y_true, y_pred):
    mre = np.abs(y_true - y_pred) / np.maximum(y_true, 1e-6)
    return np.mean(mre <= 0.25)


def evaluate_model(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MMRE': calc_mmre(y_true, y_pred),
        'Pred(25)': calc_pred25(y_true, y_pred),
    }
```

- [ ] **Step 2: 快速验证**

```bash
source .venv/bin/activate && python -c "
from core.metrics import calc_mmre, calc_pred25, evaluate_model
import numpy as np
y_true = np.array([10.0, 20.0, 30.0])
y_pred = np.array([12.0, 18.0, 33.0])
print('MMRE:', calc_mmre(y_true, y_pred))
print('Pred25:', calc_pred25(y_true, y_pred))
print('All:', evaluate_model(y_true, y_pred))
"
```

预期：打印 MMRE ≈ 0.133，Pred25 ≈ 1.0，以及完整指标字典。

- [ ] **Step 3: 提交**

```bash
git add core/metrics.py
git commit -m "feat: extract metrics module (MMRE, Pred25, evaluate_model)"
```

---

### Task 3: core/models.py — 模型训练与预测模块

**Files:**
- Create: `core/models.py`

- [ ] **Step 1: 创建 core/models.py**

从现有 `2_model_train.py` 和 `3_predict.py` 提取并封装：

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from core.metrics import evaluate_model


FEATURE_COLS = [
    'function_points', 'project_complexity', 'code_size_kloc',
    'team_experience', 'tool_maturity', 'dev_mode', 'language_type',
]

MODEL_DEFS = {
    'Linear Regression (Ridge)': {
        'model': Ridge,
        'params': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'fixed_params': {},
        'need_scale': True,
    },
    'Decision Tree': {
        'model': DecisionTreeRegressor,
        'params': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [5, 10, 15]},
        'fixed_params': {'random_state': 42},
        'need_scale': False,
    },
    'SVM (RBF)': {
        'model': SVR,
        'params': {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.1], 'epsilon': [0.1, 0.5]},
        'fixed_params': {'kernel': 'rbf'},
        'need_scale': True,
    },
    'Random Forest': {
        'model': RandomForestRegressor,
        'params': {
            'n_estimators': [100, 200, 500],
            'max_depth': [5, 10, 15, None],
            'max_features': ['sqrt', 'log2', 0.5],
            'min_samples_leaf': [1, 2, 5],
        },
        'fixed_params': {'random_state': 42},
        'need_scale': False,
    },
}

SHORT_NAMES = {
    'Linear Regression (Ridge)': 'LinReg',
    'Decision Tree': 'DTree',
    'SVM (RBF)': 'SVM',
    'Random Forest': 'RF',
}


def load_data(csv_path='data/software_projects.csv'):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y = df['actual_effort'].values
    return df, X, y


def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    trained = {}
    for name, cfg in MODEL_DEFS.items():
        model = cfg['model'](**cfg['fixed_params'])
        X_tr = X_train_scaled if cfg['need_scale'] else X_train
        grid = GridSearchCV(
            model, cfg['params'], cv=5,
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        grid.fit(X_tr, y_train)
        trained[name] = grid.best_estimator_

    return trained, scaler


def predict_effort(trained_models, scaler, params_dict):
    feature_values = [params_dict[f] for f in FEATURE_COLS]
    input_array = np.array([feature_values])
    input_scaled = scaler.transform(input_array)

    predictions = {}
    for name, model in trained_models.items():
        cfg = MODEL_DEFS[name]
        inp = input_scaled if cfg['need_scale'] else input_array
        predictions[name] = float(model.predict(inp)[0])
    return predictions


def evaluate_models(X, y, n_repeats=10, progress_callback=None):
    all_results = {name: [] for name in MODEL_DEFS}

    for i in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, cfg in MODEL_DEFS.items():
            model = cfg['model'](**cfg['fixed_params'])
            X_tr = X_train_scaled if cfg['need_scale'] else X_train
            X_te = X_test_scaled if cfg['need_scale'] else X_test
            grid = GridSearchCV(
                model, cfg['params'], cv=5,
                scoring='neg_mean_absolute_error', n_jobs=-1
            )
            grid.fit(X_tr, y_train)
            y_pred = grid.predict(X_te)
            metrics = evaluate_model(y_test, y_pred)
            all_results[name].append(metrics)

        if progress_callback:
            progress_callback(i + 1, n_repeats)

    summary = {}
    for name, results_list in all_results.items():
        avg = {}
        std = {}
        for metric in ['MAE', 'RMSE', 'MMRE', 'Pred(25)']:
            values = [r[metric] for r in results_list]
            avg[metric] = np.mean(values)
            std[metric] = np.std(values)
        summary[name] = {'avg': avg, 'std': std}
    return summary


def get_feature_importances(X, y):
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, max_features='sqrt', random_state=42)
    rf.fit(X, y)
    labels = [
        'Function Points', 'Project Complexity', 'Code Size (KLoC)',
        'Team Experience', 'Tool Maturity', 'Development Mode', 'Language Type',
    ]
    importances = rf.feature_importances_
    result = sorted(zip(labels, importances), key=lambda x: x[1], reverse=True)
    return result
```

- [ ] **Step 2: 快速验证**

```bash
source .venv/bin/activate && python -c "
from core.models import load_data, train_all_models, predict_effort
_, X, y = load_data()
models, scaler = train_all_models(X, y)
preds = predict_effort(models, scaler, {
    'function_points': 300, 'project_complexity': 3.0,
    'code_size_kloc': 50, 'team_experience': 3.5,
    'tool_maturity': 3.0, 'dev_mode': 1, 'language_type': 1,
})
for name, val in preds.items():
    print(f'{name}: {val:.2f}')
"
```

预期：4 个模型各输出一个预测值（人月），无报错。

- [ ] **Step 3: 提交**

```bash
git add core/models.py
git commit -m "feat: extract models module (train, predict, evaluate)"
```

---

### Task 4: core/database.py — 数据库模块

**Files:**
- Create: `core/database.py`

- [ ] **Step 1: 创建 core/database.py**

```python
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'estimation.db'


def _get_conn():
    return sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)


def init_db():
    conn = _get_conn()
    conn.execute('''
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
        )
    ''')
    conn.commit()
    conn.close()


def save_estimation(data: dict):
    conn = _get_conn()
    conn.execute('''
        INSERT INTO estimations (
            project_name, function_points, project_complexity,
            code_size_kloc, team_experience, tool_maturity,
            dev_mode, language_type,
            pred_ridge, pred_dtree, pred_svm, pred_rf, recommended_effort
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['project_name'],
        data['function_points'], data['project_complexity'],
        data['code_size_kloc'], data['team_experience'], data['tool_maturity'],
        data['dev_mode'], data['language_type'],
        data['pred_ridge'], data['pred_dtree'],
        data['pred_svm'], data['pred_rf'],
        data['recommended_effort'],
    ))
    conn.commit()
    conn.close()


def get_all_estimations():
    conn = _get_conn()
    df = pd.read_sql_query(
        'SELECT * FROM estimations ORDER BY created_at DESC', conn
    )
    conn.close()
    return df


def delete_estimation(est_id: int):
    conn = _get_conn()
    conn.execute('DELETE FROM estimations WHERE id = ?', (est_id,))
    conn.commit()
    conn.close()


def get_estimations_by_ids(ids: list):
    conn = _get_conn()
    placeholders = ','.join('?' * len(ids))
    df = pd.read_sql_query(
        f'SELECT * FROM estimations WHERE id IN ({placeholders})', conn, params=ids
    )
    conn.close()
    return df
```

- [ ] **Step 2: 快速验证**

```bash
source .venv/bin/activate && python -c "
import os
os.environ['DB_TEST'] = '1'
from core.database import DB_PATH, init_db, save_estimation, get_all_estimations, delete_estimation
# 使用临时数据库
import core.database as db
db.DB_PATH = '/tmp/test_estimation.db'
init_db()
save_estimation({
    'project_name': '测试项目',
    'function_points': 300, 'project_complexity': 3.0,
    'code_size_kloc': 50, 'team_experience': 3.5,
    'tool_maturity': 3.0, 'dev_mode': 1, 'language_type': 1,
    'pred_ridge': 10.5, 'pred_dtree': 11.2,
    'pred_svm': 9.8, 'pred_rf': 10.1, 'recommended_effort': 9.8,
})
df = get_all_estimations()
print(f'Records: {len(df)}')
print(df[['project_name', 'pred_svm', 'recommended_effort']])
delete_estimation(df.iloc[0]['id'])
df2 = get_all_estimations()
print(f'After delete: {len(df2)}')
os.remove('/tmp/test_estimation.db')
print('OK')
"
```

预期：插入 1 条、查询显示 1 条、删除后显示 0 条，打印 OK。

- [ ] **Step 3: 提交**

```bash
git add core/database.py
git commit -m "feat: add SQLite database module for estimation storage"
```

---

### Task 5: core/report.py — PDF 报告生成模块

**Files:**
- Create: `core/report.py`

- [ ] **Step 1: 检查系统中文字体**

```bash
fc-list :lang=zh family 2>/dev/null | head -5 || echo "fc-list not available, will use bundled font"
```

记下可用的中文字体名（macOS 通常有 STHeiti, PingFang SC 等），后续用于 fpdf2。

- [ ] **Step 2: 创建 core/report.py**

```python
import io
import os
from fpdf import FPDF
from datetime import datetime


def _find_chinese_font():
    candidates = [
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/Supplemental/Songti.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = _find_chinese_font()
        if font_path:
            self.add_font('Chinese', '', font_path, uni=True)
            self.font_name = 'Chinese'
        else:
            self.font_name = 'Helvetica'

    def header(self):
        self.set_font(self.font_name, '', 10)
        self.cell(0, 8, 'Software Project Effort Estimation Report', align='C', new_x='LMARGIN', new_y='NEXT')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_name, '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font(self.font_name, '', 14)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font(self.font_name, '', 10)
        # header row
        self.set_fill_color(76, 114, 176)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, str(h), border=1, fill=True, align='C')
        self.ln()
        # data rows
        self.set_text_color(0, 0, 0)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 8, str(val), border=1, align='C')
            self.ln()
        self.ln(4)


def generate_pdf(project_data: dict, predictions: dict, recommended_model: str):
    pdf = ReportPDF()
    pdf.add_page()

    # Title
    pdf.set_font(pdf.font_name, '', 18)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    pdf.cell(0, 12, f"Project: {project_data['project_name']}", new_x='LMARGIN', new_y='NEXT')
    pdf.set_font(pdf.font_name, '', 10)
    pdf.cell(0, 8, f"Generated: {now}", new_x='LMARGIN', new_y='NEXT')
    pdf.ln(6)

    # Project parameters
    pdf.section_title('1. Project Parameters')
    param_labels = {
        'function_points': 'Function Points',
        'project_complexity': 'Project Complexity',
        'code_size_kloc': 'Code Size (KLoC)',
        'team_experience': 'Team Experience',
        'tool_maturity': 'Tool Maturity',
        'dev_mode': 'Development Mode',
        'language_type': 'Language Type',
    }
    dev_mode_names = {0: 'Organic', 1: 'Semi-detached', 2: 'Embedded'}
    lang_type_names = {0: 'Low-level', 1: 'High-level', 2: 'Very high-level'}

    param_rows = []
    for key, label in param_labels.items():
        val = project_data[key]
        if key == 'dev_mode':
            val = f"{val} ({dev_mode_names.get(val, '')})"
        elif key == 'language_type':
            val = f"{val} ({lang_type_names.get(val, '')})"
        param_rows.append([label, str(val)])
    pdf.add_table(['Parameter', 'Value'], param_rows, [100, 90])

    # Predictions
    pdf.section_title('2. Model Predictions')
    pred_rows = []
    for model_name, value in predictions.items():
        marker = ' *' if model_name == recommended_model else ''
        pred_rows.append([model_name + marker, f"{value:.2f}"])
    pdf.add_table(['Model', 'Predicted Effort (Person-Months)'], pred_rows, [110, 80])
    pdf.set_font(pdf.font_name, '', 9)
    pdf.cell(0, 6, f"* Recommended model: {recommended_model}", new_x='LMARGIN', new_y='NEXT')
    pdf.ln(4)

    # Schedule estimation
    rec_effort = predictions[recommended_model]
    pdf.section_title('3. Schedule Estimation')
    schedule_rows = []
    for team_size in [1, 2, 3, 5]:
        months = rec_effort / team_size
        schedule_rows.append([str(team_size), f"{months:.1f}"])
    pdf.add_table(['Team Size (persons)', 'Estimated Duration (months)'], schedule_rows, [95, 95])

    # Conclusion
    pdf.section_title('4. Conclusion')
    best_team = 3 if rec_effort > 6 else 2 if rec_effort > 3 else 1
    duration = rec_effort / best_team
    pdf.set_font(pdf.font_name, '', 11)
    conclusion = (
        f"Based on {recommended_model}, the estimated effort for project "
        f"'{project_data['project_name']}' is {rec_effort:.2f} person-months. "
        f"Recommended team size: {best_team} persons, "
        f"estimated duration: {duration:.1f} months."
    )
    pdf.multi_cell(0, 7, conclusion)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.getvalue()
```

- [ ] **Step 3: 快速验证**

```bash
source .venv/bin/activate && python -c "
from core.report import generate_pdf
pdf_bytes = generate_pdf(
    project_data={
        'project_name': 'Test Project',
        'function_points': 300, 'project_complexity': 3.0,
        'code_size_kloc': 50, 'team_experience': 3.5,
        'tool_maturity': 3.0, 'dev_mode': 1, 'language_type': 1,
    },
    predictions={
        'Linear Regression (Ridge)': 10.5,
        'Decision Tree': 11.2,
        'SVM (RBF)': 9.8,
        'Random Forest': 10.1,
    },
    recommended_model='SVM (RBF)',
)
print(f'PDF size: {len(pdf_bytes)} bytes')
with open('/tmp/test_report.pdf', 'wb') as f:
    f.write(pdf_bytes)
print('Written to /tmp/test_report.pdf')
"
```

预期：打印 PDF 大小（几千字节），文件可用 PDF 阅读器打开查看。

- [ ] **Step 4: 提交**

```bash
git add core/report.py
git commit -m "feat: add PDF report generation module"
```

---

### Task 6: app.py — Streamlit 首页

**Files:**
- Create: `app.py`

- [ ] **Step 1: 创建 app.py**

```python
import streamlit as st
import pandas as pd
from core.database import init_db

init_db()

st.set_page_config(
    page_title='软件项目工作量估算系统',
    page_icon='📊',
    layout='wide',
)

st.title('软件项目工作量估算系统')
st.markdown('基于机器学习的软件项目工作量估算研究与应用 — 演示系统')

st.markdown('---')

st.subheader('系统功能')
col1, col2 = st.columns(2)
with col1:
    st.markdown('''
    - **数据探索** — 数据集统计分析与可视化
    - **模型训练** — 4 种机器学习模型训练与对比
    - **工作量预测** — 输入项目参数，生成估算报告
    ''')
with col2:
    st.markdown('''
    - **特征分析** — 特征重要性排名与分析
    - **历史记录** — 历史估算数据管理与多项目对比
    ''')

st.markdown('---')

st.subheader('数据集概况')
df = pd.read_csv('data/software_projects.csv')
c1, c2, c3, c4 = st.columns(4)
c1.metric('样本数量', f'{len(df)} 条')
c2.metric('特征数量', '7 个')
c3.metric('工作量均值', f'{df["actual_effort"].mean():.2f} 人月')
c4.metric('工作量中位数', f'{df["actual_effort"].median():.2f} 人月')
```

- [ ] **Step 2: 创建 pages 目录**

```bash
mkdir -p pages
```

- [ ] **Step 3: 运行验证**

```bash
source .venv/bin/activate && streamlit run app.py --server.headless true &
sleep 3 && curl -s http://localhost:8501 | head -20
kill %1 2>/dev/null
```

预期：curl 返回 HTML 内容，说明 Streamlit 服务启动成功。

- [ ] **Step 4: 提交**

```bash
git add app.py
git commit -m "feat: add Streamlit home page with system overview"
```

---

### Task 7: pages/1_数据探索.py

**Files:**
- Create: `pages/1_数据探索.py`

- [ ] **Step 1: 创建页面文件**

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title='数据探索', layout='wide')
st.title('数据探索与预处理')

df = pd.read_csv('data/software_projects.csv')

# 基本统计
st.subheader('基本统计信息')
st.dataframe(df.describe().round(2), use_container_width=True)

missing = df.isnull().sum()
if missing.sum() == 0:
    st.success('数据集无缺失值')
else:
    st.warning(f'缺失值: {missing[missing > 0].to_dict()}')

# 工作量分布直方图
st.subheader('工作量分布')
fig = px.histogram(df, x='actual_effort', nbins=30, color_discrete_sequence=['steelblue'])
fig.add_vline(x=df['actual_effort'].mean(), line_dash='dash', line_color='red',
              annotation_text=f"Mean={df['actual_effort'].mean():.1f}")
fig.add_vline(x=df['actual_effort'].median(), line_dash='dash', line_color='orange',
              annotation_text=f"Median={df['actual_effort'].median():.1f}")
fig.update_layout(xaxis_title='Actual Effort (Person-Months)', yaxis_title='Frequency')
st.plotly_chart(fig, use_container_width=True)

# 散点图
st.subheader('各特征与工作量的关系')
features = ['function_points', 'project_complexity', 'code_size_kloc',
            'team_experience', 'tool_maturity', 'dev_mode', 'language_type']
tabs = st.tabs(features)
for tab, feat in zip(tabs, features):
    with tab:
        fig = px.scatter(df, x=feat, y='actual_effort', opacity=0.6,
                         color_discrete_sequence=['steelblue'])
        fig.update_layout(xaxis_title=feat, yaxis_title='actual_effort')
        st.plotly_chart(fig, use_container_width=True)

# 相关性热力图
st.subheader('特征相关性热力图')
corr = df.corr().round(2)
fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1, aspect='auto')
fig.update_layout(width=700, height=600)
st.plotly_chart(fig, use_container_width=True)

# 箱线图
st.subheader('箱线图（异常值检测）')
numeric_cols = ['function_points', 'project_complexity', 'code_size_kloc',
                'team_experience', 'tool_maturity', 'actual_effort']
selected = st.multiselect('选择要查看的特征', numeric_cols, default=numeric_cols)
if selected:
    fig = go.Figure()
    for col in selected:
        fig.add_trace(go.Box(y=df[col], name=col))
    fig.update_layout(yaxis_title='Value')
    st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: 提交**

```bash
git add pages/1_数据探索.py
git commit -m "feat: add data exploration page with Plotly charts"
```

---

### Task 8: pages/2_模型训练.py

**Files:**
- Create: `pages/2_模型训练.py`

- [ ] **Step 1: 创建页面文件**

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.models import load_data, train_all_models, evaluate_models, MODEL_DEFS, SHORT_NAMES

st.set_page_config(page_title='模型训练', layout='wide')
st.title('模型训练与对比')

_, X, y = load_data()

# 训练模型（用于预测页面）
st.subheader('模型训练')
if st.button('训练模型（用于预测）', type='primary'):
    with st.spinner('正在训练 4 个模型...'):
        trained, scaler = train_all_models(X, y)
        st.session_state['trained_models'] = trained
        st.session_state['scaler'] = scaler
    st.success('模型训练完成，可前往「工作量预测」页面进行预测。')

if 'trained_models' in st.session_state:
    st.info('模型已训练，可直接使用。')

st.markdown('---')

# 10 次重复实验对比
st.subheader('模型性能对比（10 次重复实验）')
if st.button('开始评估实验'):
    progress_bar = st.progress(0)
    status = st.empty()

    def on_progress(current, total):
        progress_bar.progress(current / total)
        status.text(f'第 {current}/{total} 次实验完成')

    with st.spinner('正在进行 10 次重复实验...'):
        summary = evaluate_models(X, y, n_repeats=10, progress_callback=on_progress)

    status.text('实验完成！')

    # 结果表格
    rows = []
    for name in MODEL_DEFS:
        avg = summary[name]['avg']
        rows.append({
            '模型': name,
            'MAE': f"{avg['MAE']:.2f}",
            'RMSE': f"{avg['RMSE']:.2f}",
            'MMRE': f"{avg['MMRE']:.3f}",
            'Pred(25)': f"{avg['Pred(25)']:.3f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 对比柱状图
    metrics = ['MAE', 'RMSE', 'MMRE', 'Pred(25)']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    model_names = list(MODEL_DEFS.keys())
    short = [SHORT_NAMES[n] for n in model_names]

    cols = st.columns(4)
    for idx, metric in enumerate(metrics):
        values = [summary[m]['avg'][metric] for m in model_names]
        stds = [summary[m]['std'][metric] for m in model_names]
        fig = go.Figure(go.Bar(
            x=short, y=values, marker_color=colors,
            error_y=dict(type='data', array=stds, visible=True),
            text=[f'{v:.3f}' for v in values], textposition='outside',
        ))
        fig.update_layout(title=metric, yaxis_title=metric, height=350)
        with cols[idx]:
            st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: 提交**

```bash
git add pages/2_模型训练.py
git commit -m "feat: add model training page with evaluation and charts"
```

---

### Task 9: pages/3_工作量预测.py — 核心页面

**Files:**
- Create: `pages/3_工作量预测.py`

- [ ] **Step 1: 创建页面文件**

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.models import load_data, train_all_models, predict_effort, MODEL_DEFS, SHORT_NAMES
from core.database import save_estimation
from core.report import generate_pdf

st.set_page_config(page_title='工作量预测', layout='wide')
st.title('工作量预测')

DEV_MODE_OPTIONS = {'有机型': 0, '半分离型': 1, '嵌入型': 2}
LANG_TYPE_OPTIONS = {'低级语言': 0, '高级语言': 1, '超高级语言': 2}
RECOMMENDED_MODEL = 'SVM (RBF)'


def ensure_models():
    if 'trained_models' not in st.session_state:
        with st.spinner('首次使用，正在训练模型...'):
            _, X, y = load_data()
            trained, scaler = train_all_models(X, y)
            st.session_state['trained_models'] = trained
            st.session_state['scaler'] = scaler


def render_input_form(prefix, default_name=''):
    name = st.text_input('项目名称', value=default_name, key=f'{prefix}_name')
    c1, c2 = st.columns(2)
    with c1:
        fp = st.slider('功能点计数', 50, 1500, 300, key=f'{prefix}_fp')
        complexity = st.slider('项目复杂度', 1.0, 5.0, 3.0, 0.1, key=f'{prefix}_cx')
        code_size = st.slider('代码规模 (KLoC)', 5, 500, 50, key=f'{prefix}_cs')
        dev_mode_label = st.selectbox('开发模式', list(DEV_MODE_OPTIONS.keys()), key=f'{prefix}_dm')
    with c2:
        team_exp = st.slider('团队经验', 1.0, 5.0, 3.5, 0.1, key=f'{prefix}_te')
        tool_mat = st.slider('工具成熟度', 1.0, 5.0, 3.0, 0.1, key=f'{prefix}_tm')
        lang_label = st.selectbox('语言类型', list(LANG_TYPE_OPTIONS.keys()), key=f'{prefix}_lt')

    return {
        'project_name': name,
        'function_points': fp,
        'project_complexity': complexity,
        'code_size_kloc': code_size,
        'team_experience': team_exp,
        'tool_maturity': tool_mat,
        'dev_mode': DEV_MODE_OPTIONS[dev_mode_label],
        'language_type': LANG_TYPE_OPTIONS[lang_label],
    }


def render_report(params, predictions):
    st.markdown(f"#### {params['project_name']}")

    # 参数汇总
    st.markdown('**项目参数**')
    param_df = pd.DataFrame([{
        '功能点': params['function_points'],
        '复杂度': params['project_complexity'],
        '代码规模(KLoC)': params['code_size_kloc'],
        '团队经验': params['team_experience'],
        '工具成熟度': params['tool_maturity'],
        '开发模式': params['dev_mode'],
        '语言类型': params['language_type'],
    }])
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    # 预测结果
    st.markdown('**模型预测结果**')
    pred_rows = []
    for name, val in predictions.items():
        pred_rows.append({
            '模型': name,
            '预测工作量 (人月)': f'{val:.2f}',
            '推荐': '★' if name == RECOMMENDED_MODEL else '',
        })
    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

    # 工期换算
    rec_effort = predictions[RECOMMENDED_MODEL]
    st.markdown('**工期与人力换算**')
    schedule = []
    for size in [1, 2, 3, 5]:
        schedule.append({'团队规模 (人)': size, '预计工期 (月)': f'{rec_effort / size:.1f}'})
    st.dataframe(pd.DataFrame(schedule), use_container_width=True, hide_index=True)

    # 柱状图
    model_names = list(predictions.keys())
    short = [SHORT_NAMES[n] for n in model_names]
    values = [predictions[n] for n in model_names]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    fig = go.Figure(go.Bar(
        x=short, y=values, marker_color=colors,
        text=[f'{v:.2f}' for v in values], textposition='outside',
    ))
    fig.update_layout(yaxis_title='Predicted Effort (Person-Months)', height=350)
    st.plotly_chart(fig, use_container_width=True)

    # 结论
    best_team = 3 if rec_effort > 6 else 2 if rec_effort > 3 else 1
    duration = rec_effort / best_team
    st.info(
        f"基于 {RECOMMENDED_MODEL} 模型，项目「{params['project_name']}」"
        f"预计工作量为 **{rec_effort:.2f} 人月**。"
        f"建议 {best_team} 人团队，工期约 {duration:.1f} 个月。"
    )
    return rec_effort


# Main
ensure_models()

add_compare = st.checkbox('添加对比项目（同时评估两个项目）')

if add_compare:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader('项目 A')
        params_a = render_input_form('a', '项目 A')
    with col_b:
        st.subheader('项目 B')
        params_b = render_input_form('b', '项目 B')
else:
    params_a = render_input_form('a', '新项目')
    params_b = None

if st.button('开始估算', type='primary'):
    models = st.session_state['trained_models']
    scaler = st.session_state['scaler']

    preds_a = predict_effort(models, scaler, params_a)

    if params_b:
        preds_b = predict_effort(models, scaler, params_b)
        col_a, col_b = st.columns(2)
        with col_a:
            rec_a = render_report(params_a, preds_a)
        with col_b:
            rec_b = render_report(params_b, preds_b)

        # 对比汇总
        st.markdown('---')
        st.subheader('项目对比')
        compare_df = pd.DataFrame({
            '指标': ['推荐工作量 (人月)'],
            params_a['project_name']: [f"{rec_a:.2f}"],
            params_b['project_name']: [f"{rec_b:.2f}"],
            '差异': [f"{abs(rec_a - rec_b):.2f}"],
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.session_state['last_predictions'] = [(params_a, preds_a), (params_b, preds_b)]
    else:
        render_report(params_a, preds_a)
        st.session_state['last_predictions'] = [(params_a, preds_a)]

    st.markdown('---')

    # 保存和下载按钮
    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button('保存到数据库'):
            for params, preds in st.session_state['last_predictions']:
                save_estimation({
                    **{k: params[k] for k in [
                        'project_name', 'function_points', 'project_complexity',
                        'code_size_kloc', 'team_experience', 'tool_maturity',
                        'dev_mode', 'language_type',
                    ]},
                    'pred_ridge': preds['Linear Regression (Ridge)'],
                    'pred_dtree': preds['Decision Tree'],
                    'pred_svm': preds['SVM (RBF)'],
                    'pred_rf': preds['Random Forest'],
                    'recommended_effort': preds[RECOMMENDED_MODEL],
                })
            st.success('已保存到数据库！')

    with btn_cols[1]:
        for params, preds in st.session_state.get('last_predictions', []):
            pdf_bytes = generate_pdf(params, preds, RECOMMENDED_MODEL)
            st.download_button(
                f"下载 PDF — {params['project_name']}",
                data=pdf_bytes,
                file_name=f"estimation_{params['project_name']}.pdf",
                mime='application/pdf',
            )
```

- [ ] **Step 2: 提交**

```bash
git add pages/3_工作量预测.py
git commit -m "feat: add effort prediction page with report, DB save, PDF export"
```

---

### Task 10: pages/4_特征分析.py

**Files:**
- Create: `pages/4_特征分析.py`

- [ ] **Step 1: 创建页面文件**

```python
import streamlit as st
import plotly.express as px
import pandas as pd
from core.models import load_data, get_feature_importances

st.set_page_config(page_title='特征分析', layout='wide')
st.title('特征重要性分析')

_, X, y = load_data()
importances = get_feature_importances(X, y)

# 表格
st.subheader('特征重要性排名')
imp_df = pd.DataFrame(importances, columns=['特征', '重要性得分'])
imp_df['重要性得分'] = imp_df['重要性得分'].round(4)
st.dataframe(imp_df, use_container_width=True, hide_index=True)

# 水平条形图（升序排列，最重要的在最上面由 Plotly 自动处理 orientation='h'）
st.subheader('特征重要性可视化')
imp_df_sorted = imp_df.sort_values('重要性得分', ascending=True)
fig = px.bar(imp_df_sorted, x='重要性得分', y='特征', orientation='h',
             color='重要性得分', color_continuous_scale='RdYlGn',
             text=imp_df_sorted['重要性得分'].apply(lambda x: f'{x:.3f}'))
fig.update_layout(height=400, showlegend=False, yaxis_title='', xaxis_title='Feature Importance Score')
fig.update_traces(textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# 分析结论
top3 = imp_df.head(3)
st.subheader('分析结论')
st.info(
    f"特征重要性分析显示，影响软件项目工作量的前三大因素为："
    f"**{top3.iloc[0]['特征']}**（{top3.iloc[0]['重要性得分']:.3f}）、"
    f"**{top3.iloc[1]['特征']}**（{top3.iloc[1]['重要性得分']:.3f}）、"
    f"**{top3.iloc[2]['特征']}**（{top3.iloc[2]['重要性得分']:.3f}）。"
    f"其中 {top3.iloc[0]['特征']} 的影响最为显著。"
)
```

- [ ] **Step 2: 提交**

```bash
git add pages/4_特征分析.py
git commit -m "feat: add feature importance analysis page"
```

---

### Task 11: pages/5_历史记录.py

**Files:**
- Create: `pages/5_历史记录.py`

- [ ] **Step 1: 创建页面文件**

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.database import get_all_estimations, delete_estimation, get_estimations_by_ids

st.set_page_config(page_title='历史记录', layout='wide')
st.title('历史估算记录')

df = get_all_estimations()

if df.empty:
    st.info('暂无历史记录。请先在「工作量预测」页面进行估算并保存。')
    st.stop()

# 筛选
st.subheader('筛选')
col1, col2 = st.columns(2)
with col1:
    name_filter = st.text_input('按项目名称筛选')
with col2:
    if 'created_at' in df.columns and not df['created_at'].isna().all():
        df['created_at'] = pd.to_datetime(df['created_at'])

filtered = df.copy()
if name_filter:
    filtered = filtered[filtered['project_name'].str.contains(name_filter, case=False, na=False)]

# 展示表格
st.subheader(f'记录列表（共 {len(filtered)} 条）')
display_cols = ['id', 'project_name', 'created_at', 'function_points',
                'project_complexity', 'pred_svm', 'pred_rf', 'recommended_effort']
available_cols = [c for c in display_cols if c in filtered.columns]
st.dataframe(filtered[available_cols], use_container_width=True, hide_index=True)

# 多项目对比
st.subheader('多项目对比')
if len(filtered) >= 2:
    selected_ids = st.multiselect(
        '选择要对比的记录（按 ID）',
        filtered['id'].tolist(),
        default=filtered['id'].tolist()[:2],
    )
    if len(selected_ids) >= 2:
        compare_df = get_estimations_by_ids(selected_ids)
        model_cols = {'pred_ridge': 'LinReg', 'pred_dtree': 'DTree',
                      'pred_svm': 'SVM', 'pred_rf': 'RF'}
        fig = go.Figure()
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
        for i, (_, row) in enumerate(compare_df.iterrows()):
            values = [row[col] for col in model_cols.keys()]
            fig.add_trace(go.Bar(
                name=row['project_name'],
                x=list(model_cols.values()),
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='outside',
            ))
        fig.update_layout(
            barmode='group', yaxis_title='Predicted Effort (Person-Months)',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.caption('至少需要 2 条记录才能进行对比。')

# 删除功能
st.subheader('管理')
del_id = st.number_input('输入要删除的记录 ID', min_value=0, step=1)
if st.button('删除记录', type='secondary'):
    if del_id > 0:
        delete_estimation(int(del_id))
        st.success(f'已删除记录 ID={del_id}')
        st.rerun()
    else:
        st.warning('请输入有效的记录 ID')
```

- [ ] **Step 2: 提交**

```bash
git add pages/5_历史记录.py
git commit -m "feat: add history records page with multi-project comparison"
```

---

### Task 12: 端到端验证

**Files:** 无新增

- [ ] **Step 1: 启动 Streamlit 应用**

```bash
source .venv/bin/activate && streamlit run app.py
```

在浏览器中打开 http://localhost:8501

- [ ] **Step 2: 验证首页**

确认首页显示系统标题、功能列表、数据集统计（500 条、7 个特征、均值/中位数）。

- [ ] **Step 3: 验证数据探索页面**

打开「数据探索」页面，确认 4 组图表均正常渲染（直方图、散点图标签页、热力图、箱线图），无弹窗。

- [ ] **Step 4: 验证模型训练页面**

点击「训练模型」按钮，确认训练完成提示。点击「开始评估实验」，确认进度条推进、结果表格和 4 个柱状图正常。

- [ ] **Step 5: 验证工作量预测页面（单项目）**

输入项目参数，点击「开始估算」，确认报告完整（参数表、预测结果、工期换算、柱状图、结论）。点击「保存到数据库」和「下载 PDF」，确认功能正常。

- [ ] **Step 6: 验证工作量预测页面（双项目对比）**

勾选「添加对比项目」，输入两组不同参数，点击「开始估算」，确认两个报告并排展示，底部有对比汇总表。

- [ ] **Step 7: 验证特征分析页面**

确认特征重要性表格和条形图正常，分析结论文字包含前 3 特征名称。

- [ ] **Step 8: 验证历史记录页面**

确认之前保存的记录出现在表格中。选择多条记录进行对比，确认柱状图正常。测试删除功能。

- [ ] **Step 9: 最终提交**

```bash
git add -A
git status
git commit -m "feat: complete web-based effort estimation system"
```
