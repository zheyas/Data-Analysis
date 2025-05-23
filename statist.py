
from scipy.stats import chi2

import scipy.stats as stats

import numpy as np
import pandas as pd


def statistical_series_column(series, bins=6):
    """
    Формирует статистический ряд (таблицу частот) с ОДНОВРЕМЕННЫМ
    фиксированием интервалов от -1 до 1 и округлением до 3 знаков.

    bins: количество интервалов
    """
    s = series.dropna()
    # Строим строгие края, от -1 до 1
    edges = np.linspace(-1, 1, bins + 1)
    # Округляем до 3 знаков края
    edges = np.round(edges, 3)
    # Строим интервалы (pandas)
    intervals = pd.IntervalIndex.from_breaks(edges)
    # Категоризация
    cats = pd.cut(s, bins=intervals, include_lowest=True)
    freq = cats.value_counts().sort_index()
    df = pd.DataFrame({
        "Интервал": freq.index.astype(str),
        "Частота": freq.values
    })
    df.loc["Итого"] = ["", freq.sum()]
    return df


def descriptive_statistics(df, summary_row=True):
    """
    Строит расширенную таблицу описательной статистики по DataFrame df:
    Переменная, Мода, Медиана, Среднее, Дисперсия, Ст. откл., К_ковариации,
    Асимметрия, Экцесс, Сред. ошибка, Пред. ошибка, N

    Аргументы:
        df (pd.DataFrame): таблица данных
        summary_row (bool): если True, добавляет строку 'Итого' — общая сводная строка,
            где каждая характеристика является усреднением соответствующих статистик
            по всем переменным (например, среднее всех средних, среднее всех дисперсий и т.д.).

    Строка "Итого":
        - Это сводка по всей таблице.
        - В каждой ячейке строки "Итого" находится среднее соответствующего показателя
          по всем переменным (например, среднее арифметическое всех медиан).
        - Не относится ни к одной отдельной переменной, а характеризует всю совокупность переменных.
    """
    rows = []
    columns = list(df.columns)

    # подписываем X_1, X_2, ..., если последний Y – то он Y.
    row_labels = [f"X_{i+1}" for i in range(len(columns))]
    if columns[-1].upper() == "Y":
        row_labels[-1] = "Y"

    for col, label in zip(columns, row_labels):
        x = df[col].dropna()
        n = len(x)
        if n == 0:
            continue  # не добавлять пустые
        moda = x.mode()
        moda = moda.iloc[0] if not moda.empty else np.nan
        med = x.median()
        mean = x.mean()
        var = x.var(ddof=1)
        std = x.std(ddof=1)
        covar_coef = std / np.abs(mean) if mean != 0 else np.nan
        skewness = stats.skew(x, bias=False)
        kurt = stats.kurtosis(x, bias=False)
        se = std / np.sqrt(n)
        pred_err = se * 1.96  # 95% conf
        rows.append([
            label,
            round(moda, 3),
            round(med, 3),
            round(mean, 3),
            round(var, 3),
            round(std, 3),
            round(covar_coef, 3),
            round(skewness, 3),
            round(kurt, 3),
            round(se, 3),
            round(pred_err, 3),
            n
        ])

    if summary_row and len(rows) > 0:
        # ОБЩАЯ строка "Итого" — характеризует ВСЮ выборку
        # (в каждой колонке — среднее соответствующего показателя по столбцам)
        moda_means = df.mode().mean(numeric_only=True).mean()
        medians = df.median(numeric_only=True).mean()
        means = df.mean(numeric_only=True).mean()
        vars_ = df.var(ddof=1, numeric_only=True).mean()
        stds = df.std(ddof=1, numeric_only=True).mean()
        covs = np.mean([std / abs(mean) if mean != 0 else np.nan
                       for std, mean in zip(df.std(ddof=1, numeric_only=True), df.mean(numeric_only=True))])
        skews = df.apply(lambda x: stats.skew(x.dropna(), bias=False) if x.dropna().size else np.nan).mean()
        kurts = df.apply(lambda x: stats.kurtosis(x.dropna(), bias=False) if x.dropna().size else np.nan).mean()
        ses = np.mean([std / np.sqrt(len(df[col].dropna())) if len(df[col].dropna()) else 0 for col, std in zip(df.columns, df.std(ddof=1, numeric_only=True))])
        pred_errs = ses * 1.96
        N = int(df.shape[0])
        rows.append([
            "Итого",
            round(moda_means, 3),
            round(medians, 3),
            round(means, 3),
            round(vars_, 3),
round(stds, 3),
            round(covs, 3),
            round(skews, 3),
            round(kurts, 3),
            round(ses, 3),
            round(pred_errs, 3),
            N
        ])
    cols = [
        "Переменная", "Мода", "Медиана", "Среднее", "Дисперсия", "Ст. откл.",
        "К_ковариации", "Асимметрия", "Экцесс", "Сред. ошибка", "Пред. ошибка", "N"
    ]
    stats_df = pd.DataFrame(rows, columns=cols)
    return stats_df


def calculate_theoretical_variance(series):
    # Для равномерных распределений [0,1] берём 1/12
    if series.min() >= 0 and series.max() <= 1:
        return 1/12
    return None

def get_stats_table(df):
    """
    Таблица для проверки гипотезы о нормальности дисперсии с помощью критерия Пирсона
    """
    stats_list = []
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        x_mean = series.mean()
        Dv = series.max() - series.min()
        S2 = series.var(ddof=1)
        std_s = series.std(ddof=1)
        n = series.count()
        s0 = calculate_theoretical_variance(series) if std_s != 0 and n > 1 else None
        chi2_val = S2 * (n - 1) / (s0) if std_s != 0 and n > 1 and s0 is not None and s0 != 0 else None
        chi2_crit = chi2.ppf(0.95, df=n - 1) if n > 1 and std_s != 0 else None
        if std_s == 0 or n <= 1 or s0 == 0:
            hypothesis = 'Стандартное отклонение нулевое'
        else:
            hypothesis = 'Гипотеза принимается' if chi2_val < chi2_crit else 'Гипотеза отвергается'
        stats_list.append({
            'Переменная': column,
            'X̄': round(x_mean, 4),
            'Dᵥ': round(Dv, 4),
            'S²': round(S2, 4) if S2 is not None else None,
            's': round(std_s, 4) if std_s is not None else None,
            'σ² теор.': round(s0, 4) if s0 is not None else None,
            'X²': round(chi2_val, 4) if chi2_val is not None else None,
            'X² критическое': round(chi2_crit, 4) if chi2_crit is not None else None,
            'Гипотеза': hypothesis
        })
    return pd.DataFrame(stats_list)