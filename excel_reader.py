
import pandas as pd

def read_excel(filepath):
    """Чтение Excel-файла в pandas DataFrame"""
    df = pd.read_excel(filepath)
    return df
