import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os

class StatisticalOperations:
    def __init__(self, df):
        self.df = df
    
    def calculate_mean(self, column=None):
        """Calculate mean for specified column or all numeric columns"""
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
        """Calculate median for specified column or all numeric columns"""
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
        """Calculate mode for specified column or all numeric columns"""
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
        """Calculate standard deviation for specified column or all numeric columns"""
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
        """Calculate range (max - min) for specified column or all numeric columns"""
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
        """Calculate frequency counts or proportions for categorical variables"""
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
    
    def perform_clustering(self, features=None, clusters=3, topx=3):
        """Perform K-Means clustering on suitable columns"""
        df_copy = self.df.copy()
        
        # Handle gender encoding
        if 'gender' in df_copy.columns:
            gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0, 'male': 1, 'female': 0}
            df_copy['gender'] = df_copy['gender'].map(gender_map)
        
        if not features:
            # Auto-select features
            num_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            bin_obj_cols = [col for col in df_copy.select_dtypes(include=['object', 'category']).columns
                           if df_copy[col].nunique(dropna=False) == 2]
            
            # Encode binary object/category columns
            for col in bin_obj_cols:
                df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))
            
            features_list = num_cols + bin_obj_cols
            features_list = [col for col in features_list if df_copy[col].nunique(dropna=False) > 1 and not df_copy[col].isnull().all()]
            
            if not features_list:
                return {"error": "No suitable columns found for clustering."}
        else:
            features_list = [col.strip() for col in features.split(',')]
        
        X = df_copy[features_list].dropna()
        if X.empty:
            return {"error": "No data available after removing NaN values."}
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Profile clusters
        cluster_profiles = X.copy()
        cluster_profiles['Cluster'] = cluster_labels
        means = cluster_profiles.groupby('Cluster').mean()
        sizes = cluster_profiles['Cluster'].value_counts().sort_values(ascending=False)
        
        # Calculate distinctness
        distinctness = means.apply(lambda row: np.sum(np.abs(row)), axis=1)
        top_clusters = distinctness.sort_values(ascending=False).head(topx).index.tolist()
        
        cluster_info = {}
        for c in top_clusters:
            cluster_info[c] = {
                "size": int(sizes[c]),
                "means": means.loc[c].to_dict(),
                "distinctness": float(distinctness[c])
            }
        
        return {
            "clusters": clusters,
            "features": features_list,
            "top_clusters": cluster_info,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_sizes": sizes.to_dict()
        }

def save_results_to_json(results, operation_type, filename="operations.json"):
    """Save operation results to JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {"results": []}
    
    result_entry = {
        "operation": operation_type,
        "timestamp": pd.Timestamp.now().isoformat(),
        "data": results
    }
    
    data["results"].append(result_entry)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filename