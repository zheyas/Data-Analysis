import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t

def ols_model(norm_df):
    X = norm_df.iloc[:, :-1]
    y = norm_df.iloc[:, -1]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def regression_equation(norm_df, feature_names=None):
    """
    Возвращает строку уравнения регрессии и R².
    """
    model = ols_model(norm_df)
    params = model.params
    if feature_names is None:
        feature_names = [f"X_{i+1}" for i in range(len(params)-1)]
    terms = [f"{params.iloc[idx]:+0.4f}·{name}" for idx, name in enumerate(feature_names, start=1)]
    eq = f"Y = {params['const']:.4f} " + " ".join(terms)
    eq += f"\nКоэффициент детерминации: R² = {model.rsquared:.3f}"
    return eq

def model_adequacy(norm_df):
    """
    F-критерий Фишера, p-value и вывод об адекватности модели.
    """
    model = ols_model(norm_df)
    F = model.fvalue
    p_value = model.f_pvalue
    alpha = 0.05
    adequacy = "адекватна" if p_value < alpha else "НЕ адекватна"
    result = (
        f"F-критерий = {F:.3f}\n"
        f"p-значение = {p_value:.5f}\n"
        f"Уровень значимости: {alpha}\n"
        f"Вывод: модель {adequacy} (p < {alpha})"
    )
    return result

def analyze_coefficients(norm_df, alpha=0.05):
    """
    Возвращает DataFrame со всеми параметрами, stderr, t-стат., p, ДИ, флаг значимости и t_critical.
    """
    model = ols_model(norm_df)
    df = model.summary2().tables[1].copy()
    df.rename(columns={
        'Coef.': 'Коэффициент',
        'Std.Err.': 'Ст. ошибка',
        't': 't-стат.',
        'P>|t|': 'p-знач.',
        '[0.025': 'ДИ 2.5%',
        '0.975]': 'ДИ 97.5%'
    }, inplace=True)
    signif = (df['ДИ 2.5%'] > 0) | (df['ДИ 97.5%'] < 0)
    df['significant'] = signif.values.astype(int)  # 1 - значим, 0 - незначим

    for col in ['Коэффициент', 'Ст. ошибка', 't-стат.', 'p-знач.', 'ДИ 2.5%', 'ДИ 97.5%']:
        df[col] = df[col].map(lambda x: f"{float(x):.4f}")

    # t_critical расчёт
    n = norm_df.shape[0]
    k = norm_df.shape[1] - 1
    dof = n - k - 1
    t_critical = abs(t.ppf(alpha/2, dof)) if dof > 0 else np.nan

    return df, model, t_critical

def model_summary(norm_df):
    """
    Краткое описание модели: R², MAE, MSE, RMSE, dof, AIC/BIC.
    """
    model = ols_model(norm_df)
    y_true = model.model.endog
    y_pred = model.fittedvalues
    resid = y_true - y_pred
    n = len(resid)
    k = len(model.params) - 1
    mae = np.mean(np.abs(resid))
    mse = np.mean(resid ** 2)
    rmse = np.sqrt(mse)
    dof = n - (k + 1)
    return (
        f"Тип модели: OLS (метод наименьших квадратов)\n"
        f"Число коэффициентов (вкл. константу): {k+1}\n"
        f"Число степеней свободы: {dof}\n"
        f"MAE (средняя абсолютная ошибка): {mae:.4f}\n"
        f"R²: {model.rsquared:.4f}\n"
        f"MSE: {mse:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"AIC: {model.aic:.2f}, BIC: {model.bic:.2f}"
    )

def full_text_report(norm_df, feature_names=None, alpha=0.05):
    """
    Весь регрессионный отчет (рус., таблички, summary statsmodels).
    """
    if feature_names is None:
        feature_names = [f"X_{i+1}" for i in range(norm_df.shape[1]-1)]
    blocks = []
    blocks.append("=== Уравнение множественной линейной регрессии ===")
    blocks.append(regression_equation(norm_df, feature_names=feature_names))
    blocks.append("")
    blocks.append("=== Краткие статистики модели ===")
    blocks.append(model_summary(norm_df))
    blocks.append("")
    blocks.append("=== Проверка адекватности модели (F-критерий) ===")
    blocks.append(model_adequacy(norm_df))
    blocks.append("")
    blocks.append("===Анализ коэффициентов ===")
    df_coef, model, _ = analyze_coefficients(norm_df)
    blocks.append(df_coef.to_string())
    blocks.append("\nПояснения:\n* 'Ст. ошибка' — стандартная ошибка коэффициента\n* t-статистика — отношение значения к ошибке\n* p-значение — вероятность получить t не менее экстремальное, если коэфф. = 0\n* ДИ 2.5%/97.5% — 95% доверительный интервал\n* 'significant=True' — коэффициент статистически значим (0 не входит в ДИ)")
    blocks.append("")
    blocks.append("--- ОТЧЁТ STATS.SM OLS ---")
    blocks.append(str(model.summary()))
    return "\n".join(blocks)


def compare_models(norm_df, feature_sets):
    """
    norm_df: DataFrame с нормализованными данными
    feature_sets: список списков с именами признаков для каждой модели
    Возвращает DataFrame с метриками по всем моделям.
    """
    results = []
    y = norm_df.iloc[:, -1]
    all_features = list(norm_df.columns[:-1])

    for idx, features in enumerate(feature_sets):
        X = norm_df[features]
        X_ = sm.add_constant(X)
        model = sm.OLS(y, X_).fit()
        y_pred = model.fittedvalues
        mae = np.mean(np.abs(y - y_pred))
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        row = {
            'Модель': f"Модель {idx+1} ({', '.join(features)})",
            'Число признаков': len(features),
            'R²': round(model.rsquared, 4),
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'AIC': round(model.aic, 2),
            'BIC': round(model.bic, 2)
        }
        results.append(row)
    return pd.DataFrame(results)


def forecast(model, X_input):
    """
    X_input: список длины n (без константы!)
    """
    X_pred = np.insert(np.array(X_input), 0, 1)  # Добавить константу/intercept
    return float(np.dot(X_pred, model.params))

def comparative_analysis(norm_df, model):
    """
    Возвращает DataFrame для сравнительного анализа:
    №, Фактическое значение, Модельное значение, Погрешность, Относительная погрешность (%)
    """
    y_true = norm_df.iloc[:, -1].values
    X = norm_df.iloc[:, :-1]
    X_with_const = sm.add_constant(X)
    y_pred = model.predict(X_with_const)
    error = y_pred - y_true
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.where(y_true != 0, np.abs(error) / np.abs(y_true) * 100, np.nan)
    df = pd.DataFrame({
        "№": np.arange(1, len(y_true) + 1),
        "Фактическое значение выходного параметра": np.round(y_true, 4),
        "Модельное значение": np.round(y_pred, 4),
        "Погрешность": np.round(error, 4),
        "Относительная погрешность, %": np.round(rel_error, 2)
    })
    return df
