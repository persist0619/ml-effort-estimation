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
