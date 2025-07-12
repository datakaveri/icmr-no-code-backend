import pandas as pd
import numpy as np
import pickle

class StatisticalAnalytics:
    def __init__(self, df):
        self.df = df
    
    def calculate_mean(self, column=None):
        if column:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found in the dataframe."}
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                return {"error": f"Column '{column}' is not numeric."}
            
            mean_value = self.df[column].mean()
            return {"column": column, "mean": mean_value}
        else:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found in the dataframe."}
            
            means = self.df[numeric_cols].mean()
            return {"means": means.to_dict(), "columns": list(numeric_cols)}
    
    def calculate_median(self, column=None):
        if column:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found in the dataframe."}
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                return {"error": f"Column '{column}' is not numeric."}
            
            median_value = self.df[column].median()
            return {"column": column, "median": median_value}
        else:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found in the dataframe."}
            
            medians = self.df[numeric_cols].median()
            return {"medians": medians.to_dict(), "columns": list(numeric_cols)}
    
    def calculate_mode(self, column=None):
        def get_mode_info(col):
            mode_series = self.df[col].mode()
            if len(mode_series) == 0:
                return {"mode": None, "count": 0}
            elif len(mode_series) == 1:
                return {"mode": mode_series.iloc[0], "count": 1}
            else:
                return {"mode": list(mode_series), "count": len(mode_series)}
        
        if column:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found."}
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                return {"error": f"Column '{column}' is not numeric."}
            
            mode_info = get_mode_info(column)
            return {"column": column, **mode_info}
        else:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found in the dataframe."}
            
            modes = {}
            for col in numeric_cols:
                modes[col] = get_mode_info(col)
            
            return {"modes": modes, "columns": list(numeric_cols)}
    
    def calculate_std(self, column=None):
        if column:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found."}
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                return {"error": f"Column '{column}' is not numeric."}
            
            std_value = self.df[column].std()
            return {"column": column, "std": std_value}
        else:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found in the dataframe."}
            
            stds = self.df[numeric_cols].std()
            return {"stds": stds.to_dict(), "columns": list(numeric_cols)}
    
    def calculate_range(self, column=None):
        if column:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found in the dataframe."}
            
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                return {"error": f"Column '{column}' is not numeric."}
            
            range_value = self.df[column].max() - self.df[column].min()
            return {"column": column, "range": range_value, "min": self.df[column].min(), "max": self.df[column].max()}
        else:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return {"error": "No numeric columns found in the dataframe."}
            
            ranges = {}
            for col in numeric_cols:
                ranges[col] = {
                    "range": self.df[col].max() - self.df[col].min(),
                    "min": self.df[col].min(),
                    "max": self.df[col].max()
                }
            
            return {"ranges": ranges, "columns": list(numeric_cols)}
    
    def frequency_analysis(self, column=None, proportion=False):
        categorical_cols = self.df.select_dtypes(include=['category', 'object']).columns
        
        if column:
            if column not in self.df.columns:
                return {"error": f"Column '{column}' not found in the dataframe."}
            
            if column not in categorical_cols:
                return {"error": f"Column '{column}' is not categorical."}
            
            if proportion:
                freq = self.df[column].value_counts(normalize=True)
            else:
                freq = self.df[column].value_counts()
            
            return {"column": column, "frequency": freq.to_dict(), "proportion": proportion}
        else:
            if len(categorical_cols) == 0:
                return {"error": "No categorical columns found in the dataframe."}
            
            frequencies = {}
            for col in categorical_cols:
                if proportion:
                    freq = self.df[col].value_counts(normalize=True)
                else:
                    freq = self.df[col].value_counts()
                frequencies[col] = freq.to_dict()
            
            return {"frequencies": frequencies, "columns": list(categorical_cols), "proportion": proportion}
    
    def patient_segmentation(self, groupby_col, obs_names_path='obs_names.pkl', cond_names_path='cond_names.pkl', top_n=5):
        with open(obs_names_path, 'rb') as f:
            observation_names = pickle.load(f)
        with open(cond_names_path, 'rb') as f:
            condition_names = pickle.load(f)
        
        disease_cols = [col for col in observation_names + condition_names if col in self.df.columns]
        
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
            results["top_conditions"][group] = [(cond, group_disease.loc[group, cond], group_disease.drop(index=group)[cond].mean(), diff)
                                                for cond, diff in top_conditions.items()]
            bottom_conditions = group_diff.nsmallest(top_n)
            results["bottom_conditions"][group] = [(cond, group_disease.loc[group, cond], group_disease.drop(index=group)[cond].mean(), diff)
                                                   for cond, diff in bottom_conditions.items()]
        return results
    
    def calculate_prevalence(self, df, disease_col, case_value=1):
        valid = df[disease_col].notnull()
        total_population = valid.sum()
        n_cases = (df.loc[valid, disease_col] == case_value).sum()
    
        prevalence_prop = n_cases / total_population if total_population > 0 else float('nan')
        prevalence_pct = prevalence_prop * 100
    
        return prevalence_prop, prevalence_pct, n_cases, total_population