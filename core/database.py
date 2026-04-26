import sqlite3
import pandas as pd

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
