import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from .statistical import StatisticalAnalytics

class ClusteringAnalytics:
    def __init__(self, df):
        self.df = df

    def _is_binary(self, series):
        unique_vals = series.dropna().unique()
        return sorted(unique_vals) in ([0, 1], [False, True], [True, False])

    def _encode_gender(self, df_copy):
        if 'gender' in df_copy.columns:
            gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0, 'male': 1, 'female': 0}
            df_copy['gender'] = df_copy['gender'].map(gender_map)

    def _auto_select_features(self, df_copy):
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df_copy.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        binary_cols = [col for col in df_copy.columns if self._is_binary(df_copy[col])]
        
        cat_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        cat_cols = [col for col in cat_cols if df_copy[col].nunique(dropna=False) > 2]

        exclude_cols = set()
        for col in df_copy.columns:
            if df_copy[col].nunique(dropna=False) <= 1 or col in datetime_cols:
                exclude_cols.add(col)

        uniqueness_threshold = 20
        filtered_cols = [col for col in numeric_cols + binary_cols + cat_cols
                        if col not in exclude_cols and 
                        df_copy[col].nunique(dropna=False) <= uniqueness_threshold]

        if not filtered_cols:
            return None

        df_final = df_copy[filtered_cols].copy().dropna(axis=1, how='all')
        valid_cat_cols = [col for col in cat_cols if col in df_final.columns]
        
        if valid_cat_cols:
            df_final = pd.get_dummies(df_final, columns=valid_cat_cols, drop_first=True)
            
        return df_final

    def _manual_select_features(self, df_copy, features):
        features_list = [col.strip() for col in features.split(',')]
        missing = [col for col in features_list if col not in df_copy.columns]
        
        if missing:
            return None, f"The following features are not found in the dataset: {', '.join(missing)}"
            
        return df_copy[features_list], None

    def _determine_optimal_clusters(self, X_scaled, max_clusters=10):
        max_k = min(max_clusters, len(X_scaled))
        best_k, best_score = 2, -1
        
        for k in range(2, max_k + 1):
            try:
                kmeans_test = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans_test.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_k = k
                    best_score = score
            except Exception:
                continue
                
        return best_k

    def _build_cluster_profiles(self, df_final, cluster_labels, topx):
        cluster_profiles = df_final.copy()
        cluster_profiles['Cluster'] = cluster_labels
        
        means = cluster_profiles.groupby('Cluster').mean()
        sizes = cluster_profiles['Cluster'].value_counts().sort_values(ascending=False)
        distinctness = means.apply(lambda row: np.sum(np.abs(row)), axis=1)
        top_clusters = distinctness.sort_values(ascending=False).head(min(topx, len(means))).index.tolist()

        cluster_info = {}
        for c in top_clusters:
            cluster_info[c] = {
                "size": int(sizes[c]),
                "means": means.loc[c].to_dict(),
                "distinctness": float(distinctness[c])
            }

        return cluster_info, sizes

    def perform_clustering(self, features=None, clusters=3, topx=3, segment_clusters=False, 
                          obs_names_path=None, cond_names_path=None, top_n=5):
        df_copy = self.df.copy()
        self._encode_gender(df_copy)

        if not features:
            df_final = self._auto_select_features(df_copy)
            if df_final is None:
                return {"error": "No suitable columns found for clustering."}
        else:
            df_final, error = self._manual_select_features(df_copy, features)
            if error:
                return {"error": error}

        if df_final.empty:
            return {"error": "No data available after filtering and cleaning features."}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_final)

        if not clusters or clusters <= 0:
            clusters = self._determine_optimal_clusters(X_scaled)

        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_scaled)

        cluster_info, sizes = self._build_cluster_profiles(df_final, cluster_labels, topx)

        result = {
            "clusters": clusters,
            "features": df_final.columns.tolist(),
            "top_clusters": cluster_info,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_sizes": sizes.to_dict()
        }

        if segment_clusters and obs_names_path and cond_names_path:
            df_seg = self.df.loc[df_final.index].copy()
            df_seg['Cluster'] = cluster_labels
            seg_stats = StatisticalAnalytics(df_seg)
            segmentation_result = seg_stats.patient_segmentation(
                groupby_col='Cluster',
                obs_names_path=obs_names_path,
                cond_names_path=cond_names_path,
                top_n=top_n
            )
            result['cluster_segmentation'] = segmentation_result

        return result