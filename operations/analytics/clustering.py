import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from .statistical import StatisticalAnalytics

class ClusteringAnalytics:
    def __init__(self, df):
        self.df = df
    
    def perform_clustering(self, features=None, clusters=3, topx=3, segment_clusters=False, obs_names_path=None, cond_names_path=None, top_n=5):
        df_copy = self.df.copy()
        
        if 'gender' in df_copy.columns:
            gender_map = {'M': 1, 'F': 0, 'Male': 1, 'Female': 0, 'male': 1, 'female': 0}
            df_copy['gender'] = df_copy['gender'].map(gender_map)
        
        if not features:
            num_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            bin_obj_cols = [col for col in df_copy.select_dtypes(include=['object', 'category']).columns
                           if df_copy[col].nunique(dropna=False) == 2]
            
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
        
        #kmeans = KMeans(n_clusters=clusters, random_state=42)
        #cluster_labels = kmeans.fit_predict(X_scaled)

        silhouette_score_value = None
        if clusters is None:
            best_score = -1
            best_k = None
            best_model = None
            best_labels = None
            for k in range(2, min(11, len(X_scaled))):  # Try k from 2 to 10 or up to num_samples
                try:
                    model = KMeans(n_clusters=k, random_state=42)
                    labels = model.fit_predict(X_scaled)
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_model = model
                        best_labels = labels
                except Exception:
                    continue
            if best_model is None:
                return {"error": "Failed to determine optimal number of clusters via silhouette analysis."}
            clusters = best_k
            kmeans = best_model
            cluster_labels = best_labels
            silhouette_score_value = best_score
        else:
            kmeans = KMeans(n_clusters=clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_score_value = silhouette_score(X_scaled, cluster_labels)

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
            "cluster_sizes": sizes.to_dict(),
            "silhouette_score": float(silhouette_score_value)
        }
        
        if segment_clusters and obs_names_path and cond_names_path:
            df_seg = self.df.copy()
            df_seg = df_seg.loc[X.index].copy()
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