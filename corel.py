from scipy import stats
import numpy as np
import pandas as pd
from scipy.stats import f


def make_standard_labels(df):
    columns = list(df.columns)
    labels = [f"X_{i+1}" for i in range(len(columns)-1)] + ["Y"]
    if len(labels) != len(df.columns):
        raise ValueError(f"Length mismatch: dataframe with {len(df.columns)} columns, {len(labels)} labels")
    df_new = df.copy()
    df_new.columns = labels
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


def save_correlation_to_docx(df, filename, title="Матрица корреляций"):
    """Сохраняет матрицу в docx файл."""
    from docx import Document
    document = Document()
    document.add_heading(title, level=1)
    table = document.add_table(rows=df.shape[0] + 1, cols=df.shape[1] + 1)
    table.cell(0, 0).text = ''
    for j, col in enumerate(df.columns):
        table.cell(0, j + 1).text = str(col)
    for i, row in enumerate(df.index):
        table.cell(i + 1, 0).text = str(row)
        for j, col in enumerate(df.columns):
            cell = table.cell(i + 1, j + 1)
            val = df.iloc[i, j]
            cell.text = '' if pd.isnull(val) else str(val)
    document.save(filename)

def get_numeric_vars_only(df, exclude=['A']):
    """Оставляет только числовые столбцы (исключая, например, 'A')."""
    good = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            good.append(col)
    return df[good]

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
        # <--- Вот тут:
        plot_full_corr_pleiade(make_standard_labels(corr.iloc[:, :-1]), title="Парные корреляции (Плеяда)", top_n=top_n)
    return make_standard_labels(corr)


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
        for j in range(i+1, n):
            idx = [k for k in range(n) if k != i and k != j]
            if idx:
                Mij = np.linalg.det(R[np.ix_([i] + idx, [j] + idx)])
                Mii = np.linalg.det(R[np.ix_([i] + idx, [i] + idx)])
                Mjj = np.linalg.det(R[np.ix_([j] + idx, [j] + idx)])
                pcorr = -Mij / np.sqrt(Mii*Mjj) if abs(Mii) > 1e-10 and abs(Mjj) > 1e-10 else np.nan
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
        plot_full_corr_pleiade(make_standard_labels(p_corr), title=f"Частные корреляции (Плеяда, метод: {method})", top_n=top_n)
    return make_standard_labels(p_corr)


def plot_full_corr_pleiade(corr_matrix, title="Полный граф корреляций", top_n=5):
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.Graph()
    for col in corr_matrix.columns:
        G.add_node(col)
    all_edges = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            a = corr_matrix.columns[i]
            b = corr_matrix.columns[j]
            w = corr_matrix.iloc[i, j]
            G.add_edge(a, b, weight=abs(w))
            all_edges.append((a, b, abs(w), w))
    top_edges = sorted(all_edges, key=lambda x: -x[2])[:top_n]
    pos = nx.circular_layout(G)
    weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gold')
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='blueviolet')
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=14, font_weight='bold')
    edge_labels = { (a, b): f"{w:.2f}" for a, b, _, w in top_edges }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=10)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- Новый блок: вычислять только X_i с Y (только парные X-Y) ---
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
        result[x] = df[[x, y_col]].corr().iloc[0,1]
    return pd.DataFrame(result, index=[y_col]).T  # X_i — строки, Y — столбец