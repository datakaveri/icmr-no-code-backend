import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import itertools

class CorrelationAnalytics:
    def __init__(self, df):
        self.df = df
    
    def correlation_coefficients(self, df, col1=None, col2=None):
        results = []

        def is_numeric_or_binary(series):
            if pd.api.types.is_numeric_dtype(series):
                unique_vals = series.dropna().unique()
                if len(unique_vals) <= 20:
                    return True
                return True
            return False

        if col1 and col2:
            pairs = [(col1, col2)]
        else:
            numeric_cols = [col for col in df.columns if is_numeric_or_binary(df[col])]
            pairs = list(itertools.combinations(numeric_cols, 2))

        for c1, c2 in pairs:
            data = df[[c1, c2]].dropna()
            if data.empty:
                continue
            try:
                pearson_coef, pearson_p = pearsonr(data[c1], data[c2])
                spearman_coef, spearman_p = spearmanr(data[c1], data[c2])
                results.append({
                    'col1': c1,
                    'col2': c2,
                    'pearson_coefficient': pearson_coef,
                    'pearson_pvalue': pearson_p,
                    'spearman_coefficient': spearman_coef,
                    'spearman_pvalue': spearman_p
                })
            except Exception as e:
                results.append({
                    'col1': c1,
                    'col2': c2,
                    'pearson_coefficient': None,
                    'pearson_pvalue': None,
                    'spearman_coefficient': None,
                    'spearman_pvalue': None,
                    'error': str(e)
                })
        return pd.DataFrame(results)
    
    def calculate_covariance(self, df, col1=None, col2=None):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if col1 and col2:
            cov_value = df[col1].cov(df[col2])
            return pd.DataFrame({
                'col1': [col1],
                'col2': [col2],
                'covariance': [cov_value]
            })
        else:
            pairs = list(itertools.combinations(numeric_cols, 2))
            results = []
            for c1, c2 in pairs:
                cov_value = df[c1].cov(df[c2])
                results.append({
                    'col1': c1,
                    'col2': c2,
                    'covariance': cov_value
                })
            return pd.DataFrame(results)