import pandas as pd
import numpy as np
import statist


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

    # Сохраняем нормализованные данные
    with pd.ExcelWriter('norma.xlsx', engine='openpyxl', mode='w') as writer:
        norm_df.to_excel(writer, sheet_name='Нормализованные данные', index=False)

    return norm_df


def calc_stats(df):
    """Возвращает базовые описательные статистики"""
    return df.describe().T


def save_descriptive_stats(df, filename='norma.xlsx'):
    """
    Сохраняет описательную статистику в файл norma.xlsx на отдельный лист
    """
    stats_df = statist.descriptive_statistics(df)

    # Сохраняем в существующий файл или создаем новый
    try:
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            stats_df.to_excel(writer, sheet_name='Описательная статистика', index=False)
    except FileNotFoundError:
        # Если файл не существует, создаем новый
        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            stats_df.to_excel(writer, sheet_name='Описательная статистика', index=False)

    return stats_df


def save_all_analysis(df, filename='norma.xlsx'):
    """
    Сохраняет все результаты анализа в один файл:
    - Нормализованные данные
    - Описательная статистика
    - Проверка нормальности
    - Компактные интервальные вариационные ряды для всех переменных
    """
    # Сохраняем нормализованные данные
    with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, sheet_name='Нормализованные данные', index=False)

    # Сохраняем описательную статистику
    save_descriptive_stats(df, filename)

    # Сохраняем проверку нормальности
    statist.save_normality_stats(df, filename)

    # Сохраняем компактные интервальные ряды
    statist.save_compact_interval_series(df, filename, bins=8)

    print(f"✅ Все результаты анализа сохранены в файл '{filename}'")


