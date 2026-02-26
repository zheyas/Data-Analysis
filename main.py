import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import tksheet
import os
import statist
import excel_reader
import normalizer
import corel
import regression
import forecast
import numpy as np
from scipy.stats import f

def correlation_to_color(val):
    """Цвет текста по модулю корреляции."""
    try:
        val = float(val)
    except Exception:
        return None
    abs_v = abs(val)
    if abs_v <= 0.19:
        return "#FF4444"  # токсично-красный – очень слабая связь
    elif abs_v <= 0.39:
        return "#00BFFF"  # голубой – слабая связь
    elif abs_v <= 0.59:
        return "#AA00FF"  # фиолетовый – средняя связь
    elif abs_v <= 0.79:
        return "#FFA500"  # оранжевый – сильная связь
    else:
        return "#006400"  # темно-зеленый – очень сильная связь

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Analysis Suite")
        self.df = None
        self.norm_df = None
        self.model = None

        self.file_path = None

        self.create_widgets()

    def show_help(self):
        readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
        if not os.path.isfile(readme_path):
            messagebox.showerror("Ошибка", f"Файл README.md не найден по пути:\n{readme_path}")
            return

        with open(readme_path, 'r', encoding='utf-8') as f:
            text = f.read()

        help_win = tk.Toplevel(self.root)
        help_win.title("Справка")
        help_win.geometry("700x600")
        txt = tk.Text(help_win, wrap='word')
        txt.pack(fill='both', expand=1)
        txt.insert("1.0", text)
        txt.config(state="disabled")
        # Добавим скроллбар
        scrollb = ttk.Scrollbar(help_win, command=txt.yview)
        scrollb.pack(side=tk.RIGHT, fill=tk.Y)
        txt['yscrollcommand'] = scrollb.set

    def create_widgets(self):
        self.panel = tk.Frame(self.root)
        self.panel.grid(row=0, column=0, sticky="ns", padx=10, pady=10)
        btn_help = tk.Button(self.panel, text="Справка", width=20, command=self.show_help)
        btn_help.pack(pady=(0, 5))
        pair_corr_frame = tk.Frame(self.panel)
        pair_corr_label = tk.Label(pair_corr_frame, text="Парная корреляция:")
        pair_corr_label.pack(side=tk.LEFT)
        self.pair_corr_combo = ttk.Combobox(pair_corr_frame, state="readonly", width=22,
            values=[
                "Матрица парных корреляций",
                "t-критерий",
                "Плеяда",
                "Множественная корреляция"
            ])
        self.pair_corr_combo.current(0)
        self.pair_corr_combo.pack(side=tk.LEFT, padx=5)
        self.pair_corr_combo.bind("<<ComboboxSelected>>", self.on_pair_corr_selected)
        pair_corr_frame.pack_forget()

        p_corr_frame = tk.Frame(self.panel)
        p_corr_label = tk.Label(p_corr_frame, text="Частная корреляция:")
        p_corr_label.pack(side=tk.LEFT)
        self.part_corr_combo = ttk.Combobox(p_corr_frame, state="readonly", width=18,
            values=[
                "Матрица частных корреляций",
                "t-критерий",
                "Плеяда"
            ])
        self.part_corr_combo.current(0)
        self.part_corr_combo.pack(side=tk.LEFT, padx=5)
        self.part_corr_combo.bind("<<ComboboxSelected>>", self.on_partial_corr_selected)
        p_corr_frame.pack_forget()

        part_corr_method_frame = tk.Frame(self.panel)
        part_corr_method_frame.pack(fill=tk.X, pady=(2, 5))
        tk.Label(part_corr_method_frame,
text="Метод:").pack(side=tk.LEFT)
        self.part_corr_method = tk.StringVar(value="inverse")
        self.part_corr_method_combo = ttk.Combobox(
            part_corr_method_frame, state="readonly", width=12,
            values=["inverse", "k"], textvariable=self.part_corr_method
        )
        self.part_corr_method_combo.pack(side=tk.LEFT, padx=5)

        self.corr_frames = {
            'pair': pair_corr_frame,
            'partial': p_corr_frame,
        }

        self.stats_frame = tk.Frame(self.panel)
        self.stats_widgets = {}

        btn1 = tk.Button(self.panel, text="Открыть Excel", width=20, command=self.open_excel)
        btn1.pack(pady=5)
        btn2 = tk.Button(self.panel, text="Нормализация", width=20, command=self.run_normalize, state="disabled")
        btn2.pack(pady=5)
        btn3 = tk.Button(self.panel, text="Корреляция", width=20, command=self.show_correlation, state="disabled")
        btn3.pack(pady=5)
        btn4 = tk.Button(self.panel, text="Регрессия", width=20, command=self.show_regression, state="disabled")
        btn4.pack(pady=5)
        btn5 = tk.Button(self.panel, text="Прогнозирование", width=20, command=self.predict_y, state="disabled")
        btn5.pack(pady=5)
        btn6 = tk.Button(self.panel, text="Статистика", width=20, command=self.show_stats, state="disabled")
        btn6.pack(pady=5)
        self.btns = [btn2, btn3, btn4, btn5, btn6]

        ttk.Separator(self.panel, orient='horizontal').pack(fill=tk.X, pady=8)

        self.stats_label = tk.Label(
            self.panel,
            justify='left',
            anchor='w',
            font=('Arial', 11),
            bg='#f5f5f5',
            relief='groove',
            borderwidth=2,
            padx=6,
            pady=4,
            text=""
        )
        self.stats_label.pack(fill=tk.X, pady=(0, 5))

        self.sheet_frame = tk.Frame(self.root)
        self.sheet_frame.grid(row=0, column=1, sticky="nsew")
        self.table_label = tk.Label(self.sheet_frame, text="", font=('Arial', 14, 'bold'))
        self.table_label.pack(side=tk.TOP, pady=(6, 1))

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.sheet_frame.grid_rowconfigure(1, weight=1)
        self.sheet_frame.grid_columnconfigure(0, weight=1)

    def set_matrix_labels(self, df):
        columns = list(df.columns)
        if columns[-1].upper() == "Y":
            labels = [f"X_{i + 1}" for i in range(len(columns) - 1)] + ["Y"]
        else:
            labels = [f"X_{i + 1}" for i in range(len(columns))]
        df = df.copy()
        df.index = labels
        df.columns = labels
        return df

    def hide_corr_combos(self):
        for fr in self.corr_frames.values():
            fr.pack_forget()

    def hide_stats_widgets(self):
        for widget in self.stats_widgets.values():
            widget.pack_forget()
        self.stats_frame.pack_forget()

    def open_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            self.file_path = path
            self.df = excel_reader.read_excel(path)
            self.display_df(self.df, title="Загруженные данные")
            self.btns[0].config(state="normal")
            for b in self.btns[1:]:
                b.config(state="disabled")
            self.hide_corr_combos()
            self.hide_stats_widgets()
            self.stats_label.config(text="")

    def on_partial_corr_selected(self, event):
        value = self.part_corr_combo.get()
        method = self.part_corr_method.get()
        if self.df is None:
            messagebox.showinfo("Нет данных", "Сначала загрузите данные!")
            return
        if not hasattr(self, 'file_path') or not self.file_path:
            messagebox.showinfo("Нет файла Excel", "Сначала откройте файл Excel!")
            return
        if value == "Матрица частных корреляций":
            try:
                corr = corel.partial_correlation_matrix(self.df, method=method)
                corr = corr.round(3)
                corr = self.set_matrix_labels(corr)
                self.display_df(corr, title=f"Матрица частных корреляций ({method})", colorize=True)
            except Exception as e:
                messagebox.showerror("Ошибка при расчете", str(e))
        elif value == "t-критерий":
            t_table, t_critical = corel.partial_t_criteria(self.df, method=method)
            t_table = t_table.round(3)
            t_table = self.set_matrix_labels(t_table)
            self.display_df(
                t_table,
                title=f"t-критерий ({method}, t_критическое = {t_critical:.3f})",
                t_criteria_mode=True,
                t_critical=t_critical
            )
        elif value == "Плеяда":
            pleiade = corel.partial_pleiade(self.df, show_graph=True, method=method)
            if pleiade.shape[1] == 1 and "Плеяда" in pleiade.columns[0]:
                messagebox.showinfo("Плеяда", "В данной форме плеяда не отображается как матрица.")
                return
            self.display_df(
                pleiade,
                title=f"Плеяда частных корреляций ({method})",
                colorize=True, remove_pleiad_col=True
            )

    def run_normalize(self):
        self.hide_corr_combos()
        self.hide_stats_widgets()
        if self.df is not None:
            self.norm_df = normalizer.normalize_dataframe(self.df)
            self.display_df(self.norm_df, title="Нормализованные данные")

            # Сохраняем описательную статистику автоматически
            try:
                from normalizer import save_descriptive_stats
                save_descriptive_stats(self.norm_df, 'norma.xlsx')
            except Exception as e:
                print(f"Ошибка при сохранении описательной статистики: {e}")

            for b in self.btns[1:]:
                b.config(state="normal")
            self.stats_label.config(text="")
        else:
            messagebox.showinfo('Нет данных', 'Сначала загрузите данные!')

    def show_density_plot(self):
        if self.norm_df is None:
            tk.messagebox.showinfo('Нет данных', 'Сначала выполните нормализацию!')
            return
        # Диалог выбора переменной
        var_win = tk.Toplevel(self.root)
        var_win.title("Выбор переменной для графика плотности")
        tk.Label(var_win, text="Выберите переменную:").pack(padx=10, pady=3)
        combo = ttk.Combobox(var_win, values=list(self.norm_df.columns), state="readonly")
        combo.pack(padx=10, pady=4)

        def show_plot():
            import matplotlib.pyplot as plt
            import seaborn as sns
            col = combo.get()
            if not col:
                tk.messagebox.showinfo("Ошибка", "Выберите переменную!")
                return
            data = self.norm_df[col].dropna()
            plt.figure(figsize=(6, 4))
            sns.histplot(data, bins=10, kde=True, color="#4682b4")
            plt.title(f'График плотности распределения для {col}')
            plt.xlabel(col)
            plt.ylabel("Плотность")
            plt.tight_layout()
            plt.show()
            var_win.destroy()

        tk.Button(var_win, text="Показать", command=show_plot).pack(padx=10, pady=8)

    def show_stats(self):
        self.hide_corr_combos()
        self.hide_stats_widgets()
        if self.norm_df is None:
            messagebox.showinfo('Нет нормализованных данных', 'Сначала выполните нормализацию!')
            return

        btn_kde = tk.Button(self.stats_frame, text="График плотности распределения", width=35,
                            command=self.show_density_plot)
        btn_kde.pack(pady=3)
        self.stats_widgets['btn_kde'] = btn_kde

        self.stats_frame.pack(fill=tk.X, pady=5)
        btn_norm = tk.Button(self.stats_frame, text="Проверка на нормальность распределения по Пирсону", width=35,
                             command=self.check_normality)
        btn_norm.pack(pady=3)
        btn_desc = tk.Button(self.stats_frame, text="Описательная статистика", width=35,
                             command=self.show_descriptive_stats)
        btn_desc.pack(pady=3)

        # НОВЫЙ РАЗДЕЛ: Интервальные вариационные ряды
        interval_frame = tk.Frame(self.stats_frame, relief='groove', borderwidth=2)
        interval_frame.pack(fill=tk.X, pady=8)

        tk.Label(interval_frame, text="Интервальные вариационные ряды",
                 font=('Arial', 11, 'bold')).pack(pady=2)

        btn_interval_all = tk.Button(interval_frame, text="Построить ряды для всех переменных", width=35,
                                     command=self.build_all_interval_series)
        btn_interval_all.pack(pady=2)

        # Выпадающий список для выбора конкретной переменной
        var_select_frame = tk.Frame(interval_frame)
        var_select_frame.pack(fill=tk.X, pady=2)
        tk.Label(var_select_frame, text="Переменная:").pack(side=tk.LEFT, padx=5)

        self.interval_var_combo = ttk.Combobox(var_select_frame, values=list(self.norm_df.columns),
                                               state="readonly", width=15)
        self.interval_var_combo.pack(side=tk.LEFT, padx=5)
        self.interval_var_combo.bind("<<ComboboxSelected>>", self.show_interval_series_for_var)

        btn_hist = tk.Button(interval_frame, text="Показать гистограмму", width=35,
                             command=self.show_histogram)
        btn_hist.pack(pady=2)

        # Продолжаем существующие элементы
        series_frame = tk.Frame(self.stats_frame)
        series_frame.pack(fill=tk.X, pady=(8, 2))
        lbl = tk.Label(series_frame, text="Статистические ряды (фиксированные интервалы -1..1):")
        lbl.pack(side=tk.LEFT)
        combo = ttk.Combobox(series_frame, values=list(self.norm_df.columns), state="readonly", width=18,
                             font=('Arial', 9))
        combo.pack(side=tk.LEFT, padx=5)
        combo.bind("<<ComboboxSelected>>", self.show_statistical_series)

        self.stats_widgets = {
            'btn_norm': btn_norm,
            'btn_desc': btn_desc,
            'btn_interval_all': btn_interval_all,
            'interval_var_combo': self.interval_var_combo,
            'btn_hist': btn_hist,
            'series_frame': series_frame,
            'combo': combo
        }

    def build_all_interval_series(self):
        """Строит компактные интервальные вариационные ряды для всех переменных и сохраняет в файл"""
        if self.norm_df is None:
            messagebox.showinfo('Нет данных', 'Сначала выполните нормализацию!')
            return

        try:
            from statist import save_compact_interval_series
            save_compact_interval_series(self.norm_df, 'norma.xlsx', bins=8)
            messagebox.showinfo("Успешно",
                                "Компактные интервальные вариационные ряды сохранены в файл 'norma.xlsx'\n"
                                "на листе 'Интервальные ряды'.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить ряды: {str(e)}")

    def show_interval_series_for_var(self, event):
        """Показывает компактный интервальный ряд для выбранной переменной"""
        if self.norm_df is None:
            return

        var_name = self.interval_var_combo.get()
        if not var_name:
            return

        try:
            from statist import compact_interval_series
            series_df = compact_interval_series(self.norm_df[var_name], var_name, bins=8)
            self.display_df(series_df, title=f"Интервальный вариационный ряд для {var_name}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_interval_series_for_var(self, event):
        """Показывает интервальный ряд для выбранной переменной"""
        if self.norm_df is None:
            return

        var_name = self.interval_var_combo.get()
        if not var_name:
            return

        try:
            from statist import interval_variation_series
            series_df = interval_variation_series(self.norm_df[var_name], var_name, bins=10, is_normalized=True)
            self.display_df(series_df, title=f"Интервальный вариационный ряд для {var_name}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def show_histogram(self):
        """Показывает гистограмму для выбранной переменной"""
        if self.norm_df is None:
            messagebox.showinfo('Нет данных', 'Сначала выполните нормализацию!')
            return

        var_name = self.interval_var_combo.get()
        if not var_name:
            messagebox.showinfo("Ошибка", "Выберите переменную из списка")
            return

        try:
            from statist import plot_histogram
            plot_histogram(self.norm_df[var_name], var_name, bins=10, is_normalized=True)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def check_normality(self):
        result = statist.get_stats_table(self.norm_df)
        if isinstance(result, pd.DataFrame):
            self.display_df(result, title="Проверка на нормальность по Пирсону")

            # Сохраняем в файл norma.xlsx
            try:
                from statist import save_normality_stats
                save_normality_stats(self.norm_df, 'norma.xlsx')
            except Exception as e:
                print(f"Ошибка при сохранении проверки нормальности: {e}")
        else:
            messagebox.showinfo("Проверка на нормальность", str(result))

    def show_descriptive_stats(self):
        stats = statist.descriptive_statistics(self.norm_df)
        self.display_df(stats, title="Описательная статистика")

        # Сохраняем в файл norma.xlsx
        try:
            from normalizer import save_descriptive_stats
            save_descriptive_stats(self.norm_df, 'norma.xlsx')
        except Exception as e:
            print(f"Ошибка при сохранении описательной статистики: {e}")

    def show_statistical_series(self, event):
        variable = self.stats_widgets['combo'].get()
        if variable and self.norm_df is not None:
            series = self.norm_df[variable]
            stat_row = statist.statistical_series_column(series, bins=6)
            self.display_df(stat_row, title=f"Статистический ряд по {variable}")

    def show_correlation(self):
        self.hide_corr_combos()
        self.hide_stats_widgets()
        for fr in self.corr_frames.values():
            fr.pack(fill=tk.X, pady=(5, 0))
        self.on_pair_corr_selected(None)

    def analyze_coefficients(self):
        if self.norm_df is None:
            tk.messagebox.showinfo("Нет данных", "Сначала выполните нормализацию!")
            return
        df_coef, model, t_critical = regression.analyze_coefficients(self.norm_df)
        # Округляем только ДО 4 знаков, если не округляет regression.py
        df_coef = df_coef.map(
            lambda v: f"{float(v):.4f}" if isinstance(v, (float, int, np.floating, np.integer)) else v
        )
        # highlight_h0_accept=True включает зелёный фон для строк где гипотеза принимается!
        self.display_df(
            df_coef,
            title="Коэффициенты регрессии и их значимость",
            highlight_h0_accept=True,
            t_critical=t_critical
        )
        stats_report = regression.model_summary(self.norm_df)
        self.stats_label.config(text=stats_report)
        self.model = model

    def show_comparative_analysis(self):
        if self.norm_df is None:
            tk.messagebox.showinfo("Нет данных", "Сначала выполните нормализацию!")
            return
        if self.model is None:
            _, model, _ = regression.analyze_coefficients(self.norm_df)
        else:
            model = self.model
        df = regression.comparative_analysis(self.norm_df, model)
        self.display_df(df.set_index("№"), title="Сравнительный анализ регрессионной модели")

    def show_regression(self):
        self.hide_corr_combos()
        self.hide_stats_widgets()
        if hasattr(self, 'regression_frame'):
            self.regression_frame.destroy()
        self.regression_frame = tk.Frame(self.panel)
        self.regression_frame.pack(pady=10, fill=tk.X)
        tk.Label(self.regression_frame, text="Регрессия: анализ и полные русские отчёты",
                 font=('Arial', 12, 'bold')).pack(pady=(5, 12))

        btn1 = tk.Button(self.regression_frame, text="Показать уравнение", width=32,
                         command=self.show_regression_equation)
        btn1.pack(pady=4)
        btn2 = tk.Button(self.regression_frame, text="Анализ коэффициентов", width=32,
                         command=self.analyze_coefficients)
        btn2.pack(pady=4)
        btn3 = tk.Button(self.regression_frame, text="Полный текстовый отчет", width=32,
                         command=self.show_full_regression_report)
        btn3.pack(pady=4)
        btn5 = tk.Button(self.regression_frame, text="Оценка адекватности модели", width=32,
                         command=self.model_adequacy_eval)
        btn5.pack(pady=4)
        btn_compare = tk.Button(
            self.regression_frame, text="Сравнительный анализ регрессии", width=32,
            command=self.show_comparative_analysis)
        btn_compare.pack(pady=4)
        self.regression_buttons = [btn1, btn2, btn3, btn_compare, btn5]  # учтите последовательность

        if self.norm_df is not None:
            stats_report = regression.model_summary(self.norm_df)
            self.stats_label.config(text=stats_report)
        else:
            self.stats_label.config(text="")


    def show_regression_equation(self):
        if self.norm_df is None:
            tk.messagebox.showinfo("Нет данных", "Сначала выполните нормализацию!")
            return
        cols = list(self.norm_df.columns)
        feature_names = cols[:-1]
        eqn = regression.regression_equation(self.norm_df, feature_names=feature_names)
        tk.messagebox.showinfo("Уравнение регрессии", eqn)

    def show_coefficients_table(self):
        if self.norm_df is None:
            tk.messagebox.showinfo("Нет данных", "Сначала выполните нормализацию!")
            return
        df_coef, model = regression.analyze_coefficients(self.norm_df)
        self.display_df(df_coef, title="Таблица коэффициентов")

    def show_full_regression_report(self):
        if self.norm_df is None:
            tk.messagebox.showinfo("Нет данных", "Сначала выполните нормализацию!")
            return
        cols = list(self.norm_df.columns)
        feature_names = cols[:-1]
        report = regression.full_text_report(self.norm_df, feature_names=feature_names)
        top = tk.Toplevel(self.root)
        top.title("Полный отчет по регрессии")
        txt = tk.Text(top, wrap='word', font=("Consolas", 10))
        txt.pack(fill='both', expand=1)
        txt.insert("1.0", report)
        txt["state"] = "disabled"

    def show_t_stats_matrix(self):
        if self.norm_df is None:
            tk.messagebox.showinfo("Нет данных", "Сначала выполните нормализацию!")
            return
        df_coef, t_crit, _ = regression.analyze_coefficients(self.norm_df)
        t_stats = df_coef['t-стат.'].apply(lambda x: float(str(x).replace(',', '.')))
        t_stats_row = pd.DataFrame([t_stats.values], columns=df_coef.index)
        for col in t_stats_row.columns:
            t_stats_row[col] = t_stats_row[col].map(lambda x: f'{float(x):.3f}' if str(x) != '' else '')
        self.display_df(
            t_stats_row,
            title=f"Матрица t-статистик коэффициентов (t_крит = {t_crit:.3f})"
        )

    def model_adequacy_eval(self):
        if self.norm_df is None:
            tk.messagebox.showinfo("Нет данных", "Сначала выполните нормализацию!")
            return
        report = regression.model_adequacy(self.norm_df)
        tk.messagebox.showinfo("Оценка адекватности модели", report)

    def on_pair_corr_selected(self, event):
        value = self.pair_corr_combo.get()
        if value == "Матрица парных корреляций":
            corr = corel.correlation_analysis(self.norm_df)
            corr_disp = corr.round(3)
            corr_disp = self.set_matrix_labels(corr_disp)
            self.display_df(
                corr_disp,
                title="Матрица парных корреляций",
                colorize=True
            )
            corel.save_correlation_to_docx(corr, "Matrix.docx", title="Матрица парных корреляций")
        elif value == "t-критерий":
            t_table, t_critical = corel.t_criteria_analysis(self.norm_df)
            t_table_disp = t_table.round(3)
            t_table_disp = self.set_matrix_labels(t_table_disp)
            self.display_df(
                t_table_disp,
title=f"t-критерий (t_критическое = {t_critical:.3f})",
                t_criteria_mode=True,
                t_critical=t_critical
            )
        elif value == "Плеяда":
            pleiade = corel.pleiade_analysis(self.norm_df, show_graph=True)
            pleiade_disp = pleiade.round(3)
            pleiade_disp = self.set_matrix_labels(pleiade_disp)
            self.display_df(
                pleiade_disp,
                title="Плеяда парных корреляций",
                colorize=True,
                remove_pleiad_col=True
            )
        elif value == "Множественная корреляция":
            if self.norm_df is None:
                messagebox.showinfo('Нет нормализованных данных', 'Сначала выполните нормализацию!')
                return
            table = corel.multiple_correlation_table(self.norm_df)
            table.index = [str(i+1) for i in range(len(table))]
            self.display_df(table, title="Таблица 8 – Множественная корреляция", colorize=False)

    def predict_y(self):
        self.hide_corr_combos()
        self.hide_stats_widgets()
        if self.df is not None:
            n_x = self.df.shape[1] - 1
            top = tk.Toplevel(self.root)
            top.title("Ввод X")
            entries = []
            feature_names = list(self.df.columns)[:-1]
            for i in range(n_x):
                tk.Label(top, text=f"{feature_names[i]}:").pack()
                ent = tk.Entry(top)
                ent.pack()
                entries.append(ent)

            def calc():
                try:
                    # Важно! Считывать именно в порядке feature_names
                    X_input = [float(e.get()) for e in entries]
                    # Теперь подаем исходные данные и ручной X в forecast.forecast
                    y_pred = forecast.forecast(self.df, X_input)
                    messagebox.showinfo("Результат", f"Прогнозируемое Y: {y_pred:.4f}")
                except Exception as e:
                    messagebox.showerror("Ошибка", str(e))

            tk.Button(top, text="Предсказать", command=calc).pack()
        else:
            messagebox.showinfo("Нет данных", "Сначала загрузите Excel-файл!")

    def display_df(self, df, title="", colorize=False, t_criteria_mode=False, t_critical=None, remove_pleiad_col=False,
                   highlight_h0_accept=False):
        if remove_pleiad_col:
            for col in ["Плеяда", "Плеяды"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

        if df.shape[0] == 0 or df.shape[1] == 0:
            messagebox.showinfo("Ошибка", "Нет данных для отображения!")
            return

        row_labels = [str(idx) for idx in df.index]
        columns = [""] + list(df.columns)

        def try_format(x):
            try:
                val = float(x)
                return f"{val:.3f}"
            except:
                return x

        body = df.map(try_format).values.tolist()
        data = []
        for i, lab in enumerate(row_labels):
            data.append([lab] + body[i])

        if hasattr(self, 'sheet_widget'):
            self.sheet_widget.destroy()
        if hasattr(self, 'sheet_frame'):
            self.sheet_frame.destroy()

        self.sheet_frame = tk.Frame(self.root)
        self.sheet_frame.grid(row=0, column=1, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.sheet_frame.grid_rowconfigure(1, weight=1)
        self.sheet_frame.grid_columnconfigure(0, weight=1)

        self.table_label = tk.Label(self.sheet_frame, text=title, font=('Arial', 14, 'bold'))
        self.table_label.pack(side=tk.TOP, pady=(6, 1))

        self.sheet_widget = tksheet.Sheet(
            self.sheet_frame,
            data=data,
            headers=columns,
            show_row_index=False
        )

        # Применяем цветовое выделение для корреляций
        if colorize and not t_criteria_mode:
            for i, row in enumerate(data):
                for j, val in enumerate(row):
                    if j == 0:  # пропускаем заголовок строки
                        continue
                    color = correlation_to_color(val)
                    if color:
                        self.sheet_widget.highlight_cells(row=i, column=j, fg=color, bg=None, overwrite=True)

        # Применяем выделение для незначимых коэффициентов регрессии (светло-красный фон)
        if highlight_h0_accept and t_critical is not None:
            try:
                # Ищем столбец с p-значениями
                p_col_candidates = [c for c in df.columns if 'p-знач' in str(c) or 'p' in str(c).lower()]
                if p_col_candidates:
                    p_col = p_col_candidates[0]
                    p_col_idx = df.columns.get_loc(p_col)
                    for i, row in enumerate(data):
                        # Пропускаем строку с const, если нужно
                        p_val = row[p_col_idx + 1]
                        try:
                            p_float = float(str(p_val).replace(',', '.'))
                            # Если p-значение > 0.05, коэффициент незначим - выделяем светло-красным
                            if p_float > 0.05:
                                self.sheet_widget.highlight_rows(rows=[i], bg="#FFCCCC", fg="black", overwrite=True)
                        except:
                            continue
            except Exception as e:
                print("Highlight error:", e)

        # Применяем выделение для t-критерия
        if t_criteria_mode and t_critical is not None:
            for i, row in enumerate(data):
                for j, val in enumerate(row):
                    if j == 0:
                        continue
                    try:
                        valf = float(val.replace(',', '.'))
                        if abs(valf) >= t_critical:
                            self.sheet_widget.highlight_cells(row=i, column=j, bg='lightgreen', fg='black',
                                                              overwrite=True)
                        elif pd.notna(valf):
                            self.sheet_widget.highlight_cells(row=i, column=j, bg='#FFCCCC', fg='black', overwrite=True)
                    except:
                        continue

        self.sheet_widget.pack(fill=tk.BOTH, expand=1)
        self.sheet_widget.enable_bindings((
            "single_select",
            "row_select",
            "column_width_resize",
            "arrowkeys",
            "right_click_popup_menu",
            "rc_insert_row",
            "rc_delete_row",
            "copy",
            "cut",
            "paste",
            "undo",
            "edit_cell"
        ))



if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()