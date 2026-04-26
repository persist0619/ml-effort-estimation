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
