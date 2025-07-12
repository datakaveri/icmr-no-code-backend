import pandas as pd
import numpy as np
import os
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

class PatternAnalytics:
    def __init__(self, df):
        self.df = df
    
    def symptom_pattern_analysis_analysis(self, obs_names_path='obs_names.pkl', cond_names_path='cond_names.pkl', min_support=0.1, min_confidence=0.7, min_lift=1.2, exclude_cols=None, include_network_analysis=True, export_csv=False):
        try:
            observation_names = []
            condition_names = []
            
            if os.path.exists(obs_names_path):
                with open(obs_names_path, 'rb') as f:
                    observation_names = pickle.load(f)
            
            if os.path.exists(cond_names_path):
                with open(cond_names_path, 'rb') as f:
                    condition_names = pickle.load(f)
            
            if exclude_cols is None:
                exclude_cols = []
            
            symptom_cols = []
            for col in self.df.columns:
                if col not in exclude_cols and (col in observation_names or col in condition_names):
                    symptom_cols.append(col)
            
            if not symptom_cols:
                for col in self.df.columns:
                    if col not in exclude_cols and self.df[col].dtype in ['int64', 'float64', 'bool']:
                        unique_vals = self.df[col].dropna().unique()
                        if len(unique_vals) <= 2 and (0 in unique_vals or 1 in unique_vals):
                            symptom_cols.append(col)
            
            if not symptom_cols:
                return {"error": "No suitable symptom/condition columns found for market basket analysis."}
            
            symptoms_df = self.df[symptom_cols].copy()
            
            symptoms_df = symptoms_df.astype(bool).astype(int)
            
            symptoms_df = symptoms_df[symptoms_df.sum(axis=1) > 0]
            
            if symptoms_df.empty:
                return {"error": "No valid symptom patterns found in the data."}
            
            frequent_itemsets = apriori(symptoms_df, min_support=min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                return {"error": f"No frequent itemsets found with minimum support {min_support}."}
            
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            strong_rules = rules[
                (rules['confidence'] > min_confidence) & 
                (rules['lift'] > min_lift)
            ].sort_values('confidence', ascending=False)
            
            if export_csv and not strong_rules.empty:
                csv_data = []
                for _, rule in strong_rules.iterrows():
                    antecedent = list(rule['antecedents'])[0]
                    consequent = list(rule['consequents'])[0]
                    
                    csv_data.append({
                        'source': antecedent,
                        'target': consequent,
                        'weight': rule['confidence'],
                        'support': rule['support'],
                        'lift': rule['lift'],
                        'label': f"{rule['confidence']:.0%}"
                    })
                
                import pandas as pd
                csv_df = pd.DataFrame(csv_data)
                csv_filename = 'association_rules.csv'
                csv_df.to_csv(csv_filename, index=False)
                print(f"Association rules exported to {csv_filename}")
            
            result = {
                "parameters": {
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "min_lift": min_lift,
                    "total_symptoms_analyzed": len(symptom_cols),
                    "total_patients_with_symptoms": len(symptoms_df)
                },
                "symptom_columns": symptom_cols,
                "observation_columns": [col for col in symptom_cols if col in observation_names],
                "condition_columns": [col for col in symptom_cols if col in condition_names],
                "frequent_itemsets": {
                    "count": len(frequent_itemsets),
                    "itemsets": []
                },
                "association_rules": {
                    "total_rules": len(rules),
                    "strong_rules_count": len(strong_rules),
                    "rules": []
                }
            }
            
            for _, itemset in frequent_itemsets.iterrows():
                result["frequent_itemsets"]["itemsets"].append({
                    "itemset": list(itemset['itemsets']),
                    "support": float(itemset['support'])
                })
            
            for _, rule in strong_rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                ant_obs = [item for item in antecedents if item in observation_names]
                ant_cond = [item for item in antecedents if item in condition_names]
                cons_obs = [item for item in consequents if item in observation_names]
                cons_cond = [item for item in consequents if item in condition_names]
                
                conviction_value = rule['conviction']
                if pd.isna(conviction_value) or np.isinf(conviction_value):
                    conviction_value = None
                else:
                    conviction_value = float(conviction_value)
                
                result["association_rules"]["rules"].append({
                    "antecedents": antecedents,
                    "consequents": consequents,
                    "antecedent_observations": ant_obs,
                    "antecedent_conditions": ant_cond,
                    "consequent_observations": cons_obs,
                    "consequent_conditions": cons_cond,
                    "support": float(rule['support']),
                    "confidence": float(rule['confidence']),
                    "lift": float(rule['lift']),
                    "conviction": conviction_value
                })
            
            if include_network_analysis and not strong_rules.empty:
                G = nx.DiGraph()
                
                for _, rule in strong_rules.iterrows():
                    antecedent = list(rule['antecedents'])[0]
                    consequent = list(rule['consequents'])[0]
                    
                    G.add_edge(
                        antecedent, consequent,
                        weight=float(rule['confidence']),
                        support=float(rule['support']),
                        lift=float(rule['lift']),
                        label=f"{rule['confidence']:.0%}"
                    )
                
                if G.number_of_nodes() > 0:
                    degree_centrality = nx.degree_centrality(G)
                    in_degree_centrality = nx.in_degree_centrality(G)
                    out_degree_centrality = nx.out_degree_centrality(G)
                    
                    network_nodes = []
                    for node in G.nodes():
                        node_type = "unknown"
                        if node in observation_names:
                            node_type = "observation"
                        elif node in condition_names:
                            node_type = "condition"
                        
                        network_nodes.append({
                            "node": node,
                            "type": node_type,
                            "degree_centrality": degree_centrality.get(node, 0),
                            "in_degree_centrality": in_degree_centrality.get(node, 0),
                            "out_degree_centrality": out_degree_centrality.get(node, 0)
                        })
                    
                    network_nodes.sort(key=lambda x: x["degree_centrality"], reverse=True)
                    
                    result["network_analysis"] = {
                        "total_nodes": G.number_of_nodes(),
                        "total_edges": G.number_of_edges(),
                        "nodes": network_nodes,
                        "most_central_nodes": network_nodes[:10]
                    }
            
            return result
            
        except Exception as e:
            return {"error": f"Error in market basket analysis: {str(e)}"}