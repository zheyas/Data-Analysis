
import pandas as pd
import numpy as np

def normalize_dataframe(df):
    norm_df = df.copy()
    n = norm_df.shape[1]

    # Удаляем нечисловые столбцы, если нужно — оставьте только X_i и Y
    # norm_df = norm_df.select_dtypes(include=[np.number])

    # Нормируем только числовые столбцы
    for col in norm_df.columns:
        if pd.api.types.is_numeric_dtype(norm_df[col]):
            x_min, x_max = norm_df[col].min(), norm_df[col].max()
            if x_max != x_min:
                norm_df[col] = (norm_df[col] - x_min) / (x_max - x_min)
            else:
                norm_df[col] = 0.0

    # Переименовываем: последний — Y, остальные X_1, X_2, ...
    new_cols = [f"X_{i}" for i in range(1, n)]
    new_cols.append("Y")
    norm_df.columns = new_cols
    norm_df.to_excel('norma.xlsx', index=False)
    return norm_df

def calc_stats(df):
    """Возвращает базовые описательные статистики"""
    return df.describe().T
