import pandas as pd
import statsmodels.api as sm
import regression  # предполагается, что в regression.py есть функция ols_model(df)


def forecast(df, X_input):
    """
    df      — DataFrame с исходными (сырыми) данными (ПОСЛЕДНИЙ столбец — Y)
    X_input — список X в исходных шкалах (такой же длины, как признаков в df, не считая столбца Y)

    Возвращает: прогноз в исходных единицах
    """
    # Переобучаем модель на сырых данных!
    model = regression.ols_model(df)

    # Собираем имена признаков (без целевого столбца)
    feature_names = list(df.columns[:-1])

    # Преобразуем X_input в DataFrame с нужными именами столбцов
    X_input_df = pd.DataFrame([X_input], columns=feature_names)
    X_input_df = sm.add_constant(X_input_df, has_constant='add')

    # Делаем прогноз
    y_pred = float(model.predict(X_input_df)[0])
    return y_pred
