import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel, field_validator, ConfigDict
from typing import Optional, List

class ColumnValidation(BaseModel):
    df: pd.DataFrame
    columns: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("columns")
    def validate_columns(cls, v, values):
        if not v:
            return v
        df = values.get("df")
        if df is None:
            raise ValueError("DataFrame must be provided for validation.")
        missing_cols = [col for col in v if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataframe: {missing_cols}")
        return v

    def numeric_columns(self, exclude_binary=True):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if exclude_binary:
            binary_like = [col for col in numeric_cols 
                          if set(self.df[col].dropna().unique()) <= {0, 1}]
            numeric_cols = [col for col in numeric_cols if col not in binary_like]
        return numeric_cols

    def categorical_columns(self):
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def binary_columns(self):
        binary_cols = []
        for col in self.df.columns:
            unique_vals = set(self.df[col].dropna().unique())
            if unique_vals <= {0, 1} or unique_vals <= {True, False}:
                binary_cols.append(col)
        return binary_cols

class StatisticalAnalytics:
    def __init__(self, df):
        self.df = df
        self.validator = ColumnValidation(df=df)

    def _validate_numeric_column(self, column):
        if column not in self.validator.numeric_columns():
            print(f"Skipped non-numeric or binary column: {column}")
            return False
        return True

    def _get_valid_columns(self):
        valid_cols = self.validator.numeric_columns()
        if not valid_cols:
            return None, {"error": "No numeric columns found."}
        
        skipped = [col for col in self.df.columns if col not in valid_cols]
        if skipped:
            print(f"Skipped non-numeric or binary column(s): {', '.join(skipped)}")
        return valid_cols, None

    def calculate_mean(self, column=None):
        if column:
            if not self._validate_numeric_column(column):
                return {"error": f"No valid numeric columns in: {column}"}
            return {"column": column, "mean": self.df[column].mean()}
        
        valid_cols, error = self._get_valid_columns()
        if error:
            return error
        return {"means": self.df[valid_cols].mean().to_dict()}

    def calculate_median(self, column=None):
        if column:
            if not self._validate_numeric_column(column):
                return {"error": f"No valid numeric columns in: {column}"}
            return {"column": column, "median": self.df[column].median()}
        
        valid_cols, error = self._get_valid_columns()
        if error:
            return error
        return {"medians": self.df[valid_cols].median().to_dict()}

    def _get_mode(self, col):
        m = self.df[col].mode()
        if m.empty:
            return {"mode": None, "count": 0}
        elif len(m) == 1:
            return {"mode": m[0], "count": 1}
        else:
            return {"mode": m.tolist(), "count": len(m)}

    def calculate_mode(self, column=None):
        if column:
            if not self._validate_numeric_column(column):
                return {"error": f"No valid numeric columns in: {column}"}
            return {"column": column, **self._get_mode(column)}
        
        valid_cols, error = self._get_valid_columns()
        if error:
            return error
        return {"modes": {col: self._get_mode(col) for col in valid_cols}}

    def calculate_std(self, column=None):
        if column:
            if not self._validate_numeric_column(column):
                return {"error": f"No valid numeric columns in: {column}"}
            return {"column": column, "std": self.df[column].std()}
        
        valid_cols, error = self._get_valid_columns()
        if error:
            return error
        return {"stds": self.df[valid_cols].std().to_dict()}

    def calculate_range(self, column=None):
        if column:
            if not self._validate_numeric_column(column):
                return {"error": f"No valid numeric columns in: {column}"}
            col_data = self.df[column]
            return {
                "column": column, 
                "range": col_data.max() - col_data.min(), 
                "min": col_data.min(), 
                "max": col_data.max()
            }
        
        valid_cols, error = self._get_valid_columns()
        if error:
            return error
        return {"ranges": {col: {
            "range": self.df[col].max() - self.df[col].min(),
            "min": self.df[col].min(),
            "max": self.df[col].max()
        } for col in valid_cols}}

    def frequency_analysis(self, column=None, proportion=False):
        valid_cols = self.validator.categorical_columns()
        
        if column:
            if column not in valid_cols:
                print(f"Skipped non-categorical column: {column}")
                return {"error": f"Not a categorical column: {column}"}
            freq = self.df[column].value_counts(normalize=proportion)
            return {"column": column, "frequency": freq.to_dict(), "proportion": proportion}
        
        if not valid_cols:
            return {"error": "No categorical columns found."}
        return {
            "frequencies": {
                col: self.df[col].value_counts(normalize=proportion).to_dict() 
                for col in valid_cols
            },
            "proportion": proportion
        }

    def calculate_prevalence(self, df, disease_col, case_value=1):
        valid = df[disease_col].notnull()
        total_population = valid.sum()
        n_cases = (df.loc[valid, disease_col] == case_value).sum()
        
        prevalence_prop = n_cases / total_population if total_population > 0 else float('nan')
        prevalence_pct = prevalence_prop * 100
        
        return prevalence_prop, prevalence_pct, n_cases, total_population

    def patient_segmentation(self, groupby_col, obs_names_path='obs_names.pkl', 
                           cond_names_path='cond_names.pkl', top_n=5):
        with open(obs_names_path, 'rb') as f:
            observation_names = pickle.load(f)
        with open(cond_names_path, 'rb') as f:
            condition_names = pickle.load(f)

        disease_cols = [col for col in observation_names + condition_names 
                       if col in self.validator.binary_columns()]
        df = self.df.dropna(subset=[groupby_col])
        group_disease = df.groupby(groupby_col)[disease_cols].mean()

        results = {
            "group_prevalence": group_disease.to_dict(orient='index'),
            "top_conditions": {},
            "bottom_conditions": {}
        }

        for group in group_disease.index:
            group_diff = group_disease.loc[group] - group_disease.drop(index=group).mean()
            top_conditions = group_diff.nlargest(top_n)
            bottom_conditions = group_diff.nsmallest(top_n)
            
            results["top_conditions"][group] = [
                (cond, group_disease.loc[group, cond], 
                 group_disease.drop(index=group)[cond].mean(), diff)
                for cond, diff in top_conditions.items()
            ]
            results["bottom_conditions"][group] = [
                (cond, group_disease.loc[group, cond], 
                 group_disease.drop(index=group)[cond].mean(), diff)
                for cond, diff in bottom_conditions.items()
            ]
        return results