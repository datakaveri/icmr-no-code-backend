import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import itertools
from typing import Optional, List
from pydantic import BaseModel, field_validator, ConfigDict, ValidationInfo
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype

class ColumnPairValidation(BaseModel):
    df: pd.DataFrame
    col1: Optional[str] = None
    col2: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('col1', 'col2')
    @classmethod
    def check_columns_exist_and_valid(cls, col_name, info: ValidationInfo):
        df = info.data.get('df')
        if df is None or col_name is None:
            return col_name

        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in dataframe.")

        series = df[col_name]
        if is_datetime64_any_dtype(series):
            raise ValueError(f"Column '{col_name}' is datetime and cannot be used.")

        if not (is_numeric_dtype(series) or is_bool_dtype(series)):
            raise ValueError(f"Column '{col_name}' is not numeric or binary.")

        if series.nunique(dropna=True) <= 1:
            raise ValueError(f"Column '{col_name}' has constant value and is excluded.")

        return col_name

class CorrelationAnalytics:
    def __init__(self, df):
        self.df = df

    def _is_valid_column(self, series):
        return ((is_numeric_dtype(series) or is_bool_dtype(series)) and 
                not is_datetime64_any_dtype(series) and 
                series.nunique(dropna=True) > 1)

    def _get_valid_columns(self):
        valid_cols = []
        skipped = []
        
        for col in self.df.columns:
            series = self.df[col]
            if self._is_valid_column(series):
                valid_cols.append(col)
            elif series.nunique(dropna=True) <= 1:
                skipped.append(f"{col}: constant")
            else:
                skipped.append(f"{col}: not numeric or binary")
        
        return valid_cols, skipped

    def _compute_correlation_pair(self, c1, c2):
        data = self.df[[c1, c2]].dropna()
        if data.empty:
            return None
        
        try:
            pearson_coef, pearson_p = pearsonr(data[c1], data[c2])
            spearman_coef, spearman_p = spearmanr(data[c1], data[c2])
            return {
                'col1': c1, 'col2': c2,
                'pearson_coefficient': pearson_coef, 'pearson_pvalue': pearson_p,
                'spearman_coefficient': spearman_coef, 'spearman_pvalue': spearman_p
            }
        except Exception as e:
            return {'col1': c1, 'col2': c2, 'error': str(e)}

    def correlation_coefficients(self, df, col1: Optional[str] = None, col2: Optional[str] = None):
        results = []
        
        if col1 and col2:
            try:
                ColumnPairValidation(df=self.df, col1=col1, col2=col2)
                result = self._compute_correlation_pair(col1, col2)
                if result:
                    results.append(result)
            except ValueError as e:
                print(f"Skipped: {e}")
                return pd.DataFrame()
        else:
            valid_cols, skipped = self._get_valid_columns()
            
            if len(valid_cols) < 2:
                print("No valid numeric or binary column pairs found.")
                for msg in skipped:
                    print(f"Skipped: {msg}")
                return pd.DataFrame()

            pairs = list(itertools.combinations(valid_cols, 2))
            for c1, c2 in pairs:
                result = self._compute_correlation_pair(c1, c2)
                if result:
                    results.append(result)

            if skipped:
                print("Skipped columns:")
                for msg in skipped:
                    print(f"  {msg}")

        return pd.DataFrame(results)

    def calculate_covariance(self, df, col1: Optional[str] = None, col2: Optional[str] = None):
        results = []
        
        if col1 and col2:
            try:
                ColumnPairValidation(df=self.df, col1=col1, col2=col2)
                cov_val = self.df[col1].cov(self.df[col2])
                return pd.DataFrame([{'col1': col1, 'col2': col2, 'covariance': cov_val}])
            except ValueError as e:
                print(f"Skipped: {e}")
                return pd.DataFrame()
        else:
            valid_cols = [col for col in self.df.columns if self._is_valid_column(self.df[col])]
            
            if len(valid_cols) < 2:
                print("No valid numeric or binary columns for covariance.")
                return pd.DataFrame()

            pairs = list(itertools.combinations(valid_cols, 2))
            skipped = []

            for c1, c2 in pairs:
                try:
                    cov_val = self.df[c1].cov(self.df[c2])
                    results.append({'col1': c1, 'col2': c2, 'covariance': cov_val})
                except Exception as e:
                    skipped.append(f"{c1} vs {c2}: {e}")
            
            if skipped:
                print("Skipped pairs:")
                for msg in skipped:
                    print(f"  {msg}")
                    
        return pd.DataFrame(results)