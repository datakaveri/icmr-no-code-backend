import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import pearsonr, spearmanr
import json
import os
import pickle
import itertools

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
    def patient_segmentation(self, groupby_col, obs_names_path='obs_names.pkl', cond_names_path='cond_names.pkl', top_n=5):
        """
        Generalized patient segmentation:
        - For each group in groupby_col, shows mean prevalence of all disease/observation/condition columns.
        - Highlights the top N most distinctive conditions for each group.
        """
        # Load observation and condition names
        with open(obs_names_path, 'rb') as f:
            observation_names = pickle.load(f)
        with open(cond_names_path, 'rb') as f:
            condition_names = pickle.load(f)
        
        # Only keep columns that are in the dataframe
        disease_cols = [col for col in observation_names + condition_names if col in self.df.columns]
        
        # Drop rows with missing groupby_col
        df = self.df.dropna(subset=[groupby_col])
        
        # Group by the selected column and compute mean prevalence for each group
        group_disease = df.groupby(groupby_col)[disease_cols].mean()
        results = {
            "group_prevalence": group_disease.to_dict(orient='index'),
            "top_conditions": {},
            "bottom_conditions": {}
        }
        
        # For each group, find the most distinctive conditions (compared to the mean of all other groups)
        for group in group_disease.index:
            group_diff = group_disease.loc[group] - group_disease.drop(index=group).mean()
            top_conditions = group_diff.nlargest(top_n)
            results["top_conditions"][group] = [(cond, group_disease.loc[group, cond], group_disease.drop(index=group)[cond].mean(), diff)
                                                for cond, diff in top_conditions.items()]
            bottom_conditions = group_diff.nsmallest(top_n)
            results["bottom_conditions"][group] = [(cond, group_disease.loc[group, cond], group_disease.drop(index=group)[cond].mean(), diff)
                                                   for cond, diff in bottom_conditions.items()]
        return results
    
    def perform_clustering(self, features=None, clusters=3, topx=3, segment_clusters=False, obs_names_path=None, cond_names_path=None, top_n=5):
        """Perform K-Means clustering on suitable columns, optionally segment clusters."""
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

        # After assigning cluster_labels
        cluster_profiles = X.copy()
        cluster_profiles['Cluster'] = cluster_labels
        means = cluster_profiles.groupby('Cluster').mean()
        sizes = cluster_profiles['Cluster'].value_counts().sort_values(ascending=False)
        distinctness = means.apply(lambda row: np.sum(np.abs(row)), axis=1)
        top_clusters = distinctness.sort_values(ascending=False).head(topx).index.tolist()
        cluster_info = {}
        for c in top_clusters:
            cluster_info[c] = {
                "size": int(sizes[c]),
                "means": means.loc[c].to_dict(),
                "distinctness": float(distinctness[c])
            }
        result = {
            "clusters": clusters,
            "features": features_list,
            "top_clusters": cluster_info,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_sizes": sizes.to_dict()
        }
        # --- Patient segmentation feature ---
        if segment_clusters and obs_names_path and cond_names_path:
            # Add cluster labels to the original dataframe for segmentation
            df_seg = self.df.copy()
            df_seg = df_seg.loc[X.index].copy()
            df_seg['Cluster'] = cluster_labels
            # Use patient_segmentation with 'Cluster' as groupby_col
            seg_stats = StatisticalOperations(df_seg)
            segmentation_result = seg_stats.patient_segmentation(
                groupby_col='Cluster',
                obs_names_path=obs_names_path,
                cond_names_path=cond_names_path,
                top_n=top_n
            )
            result['cluster_segmentation'] = segmentation_result
        return result

    def create_confusion_matrix(self, y_true_col, y_pred_col):
        """
        Creates a confusion matrix given the true and predicted column names in the dataframe.
        Returns the confusion matrix as a pandas DataFrame.
        """
        y_true = self.df[y_true_col]
        y_pred = self.df[y_pred_col]
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.index.name = 'Actual'
        cm_df.columns.name = 'Predicted'
        return cm_df
    
    def calculate_prevalence(self, df, disease_col, case_value=1):
        """
        Calculates the point prevalence of a disease in the dataset.

        Args:
            df (pd.DataFrame): The input DataFrame.
            disease_col (str): Column name indicating disease status (binary or categorical).
            case_value (int/str): Value in disease_col indicating a case (default: 1).

        Returns:
            float: Prevalence as a proportion (0-1).
            float: Prevalence as a percentage (0-100).
            int: Number of cases.
            int: Total population (denominator).
        """
        # Exclude missing values in the disease column
        valid = df[disease_col].notnull()
        total_population = valid.sum()
        n_cases = (df.loc[valid, disease_col] == case_value).sum()

        prevalence_prop = n_cases / total_population if total_population > 0 else float('nan')
        prevalence_pct = prevalence_prop * 100

        return prevalence_prop, prevalence_pct, n_cases, total_population

    def correlation_coefficients(self, df, col1=None, col2=None):
        """
        Calculates Pearson and Spearman correlation coefficients for specified columns,
        or for all unique pairs of numeric/binary columns if none specified.

        Args:
            df (pd.DataFrame): Input DataFrame.
            col1 (str, optional): First column name.
            col2 (str, optional): Second column name.

        Returns:
            pd.DataFrame: Results table with columns:
                ['col1', 'col2', 'pearson_coefficient', 'pearson_pvalue', 'spearman_coefficient', 'spearman_pvalue']
        """
        results = []

        # Helper: get all numeric/binary columns
        def is_numeric_or_binary(series):
            if pd.api.types.is_numeric_dtype(series):
                unique_vals = series.dropna().unique()
                if len(unique_vals) <= 20:  # Arbitrary: treat <=20 unique as binary/categorical
                    return True
                return True
            return False

        # If columns are specified, just use those
        if col1 and col2:
            pairs = [(col1, col2)]
        else:
            # Get all numeric/binary columns
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
        """
        Calculates covariance between specified columns or all numeric pairs.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            col1 (str): First column (optional)
            col2 (str): Second column (optional)
            
        Returns:
            pd.DataFrame: Columns ['col1', 'col2', 'covariance']
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if col1 and col2:
            # Single pair case
            cov_value = df[col1].cov(df[col2])
            return pd.DataFrame({
                'col1': [col1],
                'col2': [col2],
                'covariance': [cov_value]
            })
        else:
            # All pairs case
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