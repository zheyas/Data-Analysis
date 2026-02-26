from scipy import stats
import numpy as np
import pandas as pd
from scipy.stats import f


def make_standard_labels(df):
    """
    Переименовывает колонки и индексы DataFrame в формат X_1, X_2, ..., Y.
    """
    columns = list(df.columns)
    n = len(columns)

    # Создаем метки: для последнего столбца - Y, для остальных - X_i
    labels = [f"X_{i + 1}" for i in range(n - 1)] + ["Y"]

    # Проверяем соответствие длины
    if len(labels) != n:
        # Если не совпадает, пробуем другой подход
        if n == len(columns):
            # Просто используем现有的 имена
            labels = columns

    df_new = df.copy()
    df_new.columns = labels

    # Устанавливаем индексы только если количество совпадает
    if len(labels) == df_new.shape[0]:
        df_new.index = labels

    return df_new

def rename_last_to_Y(df_corr):
    cols = list(df_corr.columns)
    cols[-1] = 'Y'
    df_corr.columns = cols
    df_corr.index = cols
    return df_corr


def multiple_correlation_table(df, y_col=None, alpha=0.05):
    """Расчёт множественной корреляции и F-критерия; возвращает таблицу."""
    if y_col is None:
        y_idx = df.shape[1] - 1  # последний столбец как Y по умолчанию
    else:
        y_idx = df.columns.get_loc(y_col) if isinstance(y_col, str) else y_col
    X_cols = [i for i in range(df.shape[1]) if i != y_idx]
    k = len(X_cols)
    n = df.shape[0]
    R = df.corr().values
    D = np.linalg.det(R)
    R_xx = np.delete(np.delete(R, y_idx, axis=0), y_idx, axis=1)
    D_yy = np.linalg.det(R_xx)
    if D_yy == 0:
        R_squared = np.nan
        R_multiple = 0.0
        F_num = np.nan
    else:
        R_squared = 1 - D / D_yy
        R_multiple = np.sqrt(R_squared) if R_squared >= 0 else 0.0
        F_num = (n - k - 1) / k * R_squared / (1 - R_squared) if (1 - R_squared) != 0 else np.inf
    F_critical = f.ppf(1 - alpha, dfn=k, dfd=n - k - 1)
    signif = "значим" if (F_num > F_critical) else "не значим"
    df_out = pd.DataFrame({
        'Показатель': [
            'Определитель полной матрицы (D)',
            'Определитель подматрицы D_yy',
            'Множественный коэффициент корреляции (R)',
            'R^2',
            'F наблюдаемое',
            f'F критическое (α = {alpha:.2f})',
            'Значимость'
        ],
        'Значение': [
            D, D_yy, R_multiple, R_squared, F_num, F_critical, signif
        ]
    })
    return df_out




def save_correlation_to_excel(df, filename, sheet_name="Корреляции"):
    """Сохраняет матрицу в Excel с цветовой заливкой."""
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name)

        # Получаем рабочий лист для применения стилей
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Применяем цветовую заливку
        from openpyxl.styles import PatternFill, Font

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                cell = worksheet.cell(row=i + 2, column=j + 2)  # +2 из-за заголовков
                try:
                    val = float(cell.value)
                    abs_v = abs(val)

                    if abs_v <= 0.19:
                        fill = PatternFill(start_color="FF4444", end_color="FF4444", fill_type="solid")
                        font = Font(color="FFFFFF")  # белый текст на красном
                    elif abs_v <= 0.39:
                        fill = PatternFill(start_color="FF69B4", end_color="FF69B4", fill_type="solid")
                        font = Font(color="000000")  # черный текст на розовом
                    elif abs_v <= 0.59:
                        fill = PatternFill(start_color="AA00FF", end_color="AA00FF", fill_type="solid")
                        font = Font(color="FFFFFF")  # белый текст на фиолетовом
                    elif abs_v <= 0.79:
                        fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")
                        font = Font(color="000000")  # черный текст на светло-зеленом
                    else:
                        fill = PatternFill(start_color="006400", end_color="006400", fill_type="solid")
                        font = Font(color="FFFFFF")  # белый текст на темно-зеленом

                    cell.fill = fill
                    cell.font = font
                except:
                    pass


def get_correlation_strength(val):
    """Возвращает текстовое описание силы связи."""
    abs_v = abs(val)
    if abs_v <= 0.19:
        return "очень слабая"
    elif abs_v <= 0.39:
        return "слабая"
    elif abs_v <= 0.59:
        return "средняя"
    elif abs_v <= 0.79:
        return "сильная"
    else:
        return "очень сильная"

def correlation_analysis(norm_df):
    """Стандартная полная матрица парных корреляций Пирсона."""
    return norm_df.corr()


def t_criteria_analysis(norm_df, alpha=0.05):
    corr = norm_df.corr()
    n = norm_df.shape[0]
    dfree = n - 2
    t_critical = stats.t.ppf(1 - alpha / 2, dfree)
    t_mat = corr.copy()
    for i in corr.columns:
        for j in corr.columns:
            if i != j:
                r = corr.loc[i, j]
                t = abs(r) * np.sqrt((n - 2) / (1 - r ** 2)) if abs(r) < 1 else np.nan
                t_mat.loc[i, j] = t
            else:
                t_mat.loc[i, j] = np.nan
    return t_mat, t_critical


def pleiade_analysis(norm_df, show_graph=True, top_n=5):
    corr = correlation_analysis(norm_df).abs()
    np.fill_diagonal(corr.values, 0)
    corr['Плеяда'] = corr.sum(axis=1)
    if show_graph:
        # Берем только корреляционную матрицу без столбца 'Плеяда'
        plot_full_corr_pleiade(
            make_standard_labels(corr.iloc[:, :-1].copy()),
            title="Парные корреляции (Плеяда)",
            top_n=top_n
        )
    # Возвращаем с правильными метками
    result = corr.copy()
    result = make_standard_labels(result)
    return result


def partial_correlation_matrix_inverse(df, regularization=1e-6):
    df = df.select_dtypes(include='number').copy()
    cols = df.columns
    n = len(cols)
    R = df.corr().values.copy()
    R += np.eye(n) * regularization
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        raise ValueError('Корреляционная матрица необратима! Проверьте данные (возможно мультиколлинеарность).')
    part_corr = np.zeros_like(R)
    for i in range(n):
        for j in range(n):
            part_corr[i, j] = 1.0 if i == j else -R_inv[i, j] / np.sqrt(R_inv[i, i] * R_inv[j, j])
    part_corr_df = pd.DataFrame(part_corr, index=cols, columns=cols)
    part_corr_df = rename_last_to_Y(part_corr_df)
    return part_corr_df


def partial_correlation_matrix_k(df):
    df = df.select_dtypes(include='number').copy()
    cols = df.columns
    n = len(cols)
    R = df.corr().values
    part_corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            idx = [k for k in range(n) if k != i and k != j]
            if idx:
                Mij = np.linalg.det(R[np.ix_([i] + idx, [j] + idx)])
                Mii = np.linalg.det(R[np.ix_([i] + idx, [i] + idx)])
                Mjj = np.linalg.det(R[np.ix_([j] + idx, [j] + idx)])
                pcorr = -Mij / np.sqrt(Mii * Mjj) if abs(Mii) > 1e-10 and abs(Mjj) > 1e-10 else np.nan
            else:
                pcorr = R[i, j]
            part_corr[i, j] = part_corr[j, i] = pcorr
    part_corr_df = pd.DataFrame(part_corr, index=cols, columns=cols)
    part_corr_df = rename_last_to_Y(part_corr_df)
    return part_corr_df


def partial_correlation_matrix(df, method='inverse'):
    """
    Обобщённый вызов: частная корреляция (метод 'inverse' или 'k')
    """
    if method == 'inverse':
        return partial_correlation_matrix_inverse(df)
    elif method == 'k':
        return partial_correlation_matrix_k(df)
    else:
        raise ValueError('Unknown method for partial correlation: use "inverse" or "k"')


def partial_t_criteria(orig_df, alpha=0.05, method='inverse'):
    p_corr = partial_correlation_matrix(orig_df, method=method)
    n = orig_df.shape[0]
    p = orig_df.shape[1]
    k = p - 2
    dfree = n - k - 2
    if dfree < 1:
        raise ValueError("Недостаточно степеней свободы для t-статистики частных корреляций. Добавьте наблюдения.")
    t_critical = stats.t.ppf(1 - alpha / 2, dfree)
    t_mat = pd.DataFrame(index=p_corr.index, columns=p_corr.columns, dtype=float)
    for i in p_corr.index:
        for j in p_corr.columns:
            if i == j:
                t_mat.loc[i, j] = 0.0
            else:
                r = p_corr.loc[i, j]
                if pd.notna(r) and abs(r) < 1:
                    t = r * np.sqrt((n - k - 2) / (1 - r ** 2))
                    t_mat.loc[i, j] = t
                else:
                    t_mat.loc[i, j] = np.nan
    return t_mat, t_critical


def partial_pleiade(orig_df, show_graph=True, top_n=5, method='inverse'):
    p_corr = partial_correlation_matrix(orig_df, method=method).abs()
    np.fill_diagonal(p_corr.values, 0)
    if show_graph:
        plot_full_corr_pleiade(make_standard_labels(p_corr), title=f"Частные корреляции (Плеяда, метод: {method})",
                               top_n=top_n)
    return make_standard_labels(p_corr)


def save_correlation_to_docx(df, filename, title="Матрица корреляций"):
    """Сохраняет матрицу в docx файл с цветовой кодировкой."""
    from docx import Document
    from docx.shared import RGBColor

    document = Document()
    document.add_heading(title, level=1)

    # Создаем таблицу
    table = document.add_table(rows=df.shape[0] + 1, cols=df.shape[1] + 1)
    table.style = 'Table Grid'

    # Заполняем заголовки
    table.cell(0, 0).text = ''
    for j, col in enumerate(df.columns):
        table.cell(0, j + 1).text = str(col)

    # Заполняем данные с цветовой кодировкой
    for i, row in enumerate(df.index):
        table.cell(i + 1, 0).text = str(row)
        for j, col in enumerate(df.columns):
            cell = table.cell(i + 1, j + 1)
            val = df.iloc[i, j]

            if pd.isnull(val):
                cell.text = ''
            else:
                cell.text = f"{val:.3f}"

                # Применяем цвет в зависимости от величины корреляции
                abs_v = abs(float(val))
                if abs_v <= 0.19:
                    color = RGBColor(0xFF, 0x44, 0x44)  # токсично-красный
                elif abs_v <= 0.39:
                    color = RGBColor(0x00, 0xBF, 0xFF)  # голубой
                elif abs_v <= 0.59:
                    color = RGBColor(0xAA, 0x00, 0xFF)  # фиолетовый
                elif abs_v <= 0.79:
                    color = RGBColor(0xFF, 0xA5, 0x00)  # оранжевый
                else:
                    color = RGBColor(0x00, 0x64, 0x00)  # темно-зеленый

                # Применяем цвет к тексту ячейки
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = color

    document.save(filename)


def save_correlation_to_excel(df, filename, sheet_name="Корреляции"):
    """Сохраняет матрицу в Excel с цветовой заливкой."""
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name)

        # Получаем рабочий лист для применения стилей
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # Применяем цветовую заливку
        from openpyxl.styles import PatternFill, Font

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                cell = worksheet.cell(row=i + 2, column=j + 2)  # +2 из-за заголовков
                try:
                    val = float(cell.value)
                    abs_v = abs(val)

                    if abs_v <= 0.19:
                        fill = PatternFill(start_color="FF4444", end_color="FF4444", fill_type="solid")
                        font = Font(color="FFFFFF")  # белый текст на красном
                    elif abs_v <= 0.39:
                        fill = PatternFill(start_color="00BFFF", end_color="00BFFF", fill_type="solid")
                        font = Font(color="000000")  # черный текст на голубом
                    elif abs_v <= 0.59:
                        fill = PatternFill(start_color="AA00FF", end_color="AA00FF", fill_type="solid")
                        font = Font(color="FFFFFF")  # белый текст на фиолетовом
                    elif abs_v <= 0.79:
                        fill = PatternFill(start_color="FFA500", end_color="FFA500", fill_type="solid")
                        font = Font(color="000000")  # черный текст на оранжевом
                    else:
                        fill = PatternFill(start_color="006400", end_color="006400", fill_type="solid")
                        font = Font(color="FFFFFF")  # белый текст на темно-зеленом

                    cell.fill = fill
                    cell.font = font
                except:
                    pass


def plot_full_corr_pleiade(corr_matrix, title="Полный граф корреляций", top_n=5):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Функция для получения цвета ребра в зависимости от силы связи
    def get_edge_color(weight):
        if weight <= 0.19:
            return "#FF4444"  # токсично-красный
        elif weight <= 0.39:
            return "#00BFFF"  # голубой
        elif weight <= 0.59:
            return "#AA00FF"  # фиолетовый
        elif weight <= 0.79:
            return "#FFA500"  # оранжевый
        else:
            return "#006400"  # темно-зеленый

    G = nx.Graph()
    for col in corr_matrix.columns:
        G.add_node(col)

    all_edges = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            a = corr_matrix.columns[i]
            b = corr_matrix.columns[j]
            w = abs(corr_matrix.iloc[i, j])
            G.add_edge(a, b, weight=w)
            all_edges.append((a, b, w, corr_matrix.iloc[i, j]))

    top_edges = sorted(all_edges, key=lambda x: -x[2])[:top_n]
    pos = nx.circular_layout(G)

    # Рисуем ребра с разными цветами
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        color = get_edge_color(weight)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=weight * 5, edge_color=color)

    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='blueviolet')
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=14, font_weight='bold')

    edge_labels = {(a, b): f"{w:.2f}" for a, b, w, _ in top_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=10)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def pairwise_x_with_y(df):
    """
    DataFrame с корреляциями каждого X_i с Y (Y — последний столбец).
    """
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("В таблице должно быть минимум две переменных!")
    y_col = cols[-1]
    x_cols = cols[:-1]
    result = {}
    for x in x_cols:
        result[x] = df[[x, y_col]].corr().iloc[0, 1]
    return pd.DataFrame(result, index=[y_col]).T  # X_i — строки, Y — столбец
