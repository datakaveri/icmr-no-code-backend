import pandas as pd
import numpy as np
import os
import pickle
import networkx as nx
from typing import List, Optional
from mlxtend.frequent_patterns import apriori, association_rules
from pydantic import BaseModel, Field, field_validator

class SymptomPatternParams(BaseModel):
    obs_names_path: str = 'obs_names.pkl'
    cond_names_path: str = 'cond_names.pkl'
    min_support: float = Field(ge=0.0, le=1.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    min_lift: float = Field(gt=0.0)
    exclude_cols: Optional[List[str]] = []
    include_network_analysis: bool = True
    export_csv: bool = False

    @field_validator("obs_names_path", "cond_names_path")
    @classmethod
    def file_must_exist(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"File not found: {v}")
        return v

class PatternAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _load_mappings(self, obs_path, cond_path):
        with open(obs_path, 'rb') as f:
            observation_names = pickle.load(f)
        with open(cond_path, 'rb') as f:
            condition_names = pickle.load(f)
        return observation_names, condition_names

    def _identify_symptom_columns(self, params, observation_names, condition_names):
        symptom_cols = [
            col for col in self.df.columns
            if col not in params.exclude_cols and 
            (col in observation_names or col in condition_names)
        ]

        if not symptom_cols:
            for col in self.df.columns:
                if col not in params.exclude_cols and self.df[col].dtype in ['int64', 'float64', 'bool']:
                    unique_vals = self.df[col].dropna().unique()
                    if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                        symptom_cols.append(col)
        
        return symptom_cols

    def _prepare_symptoms_dataframe(self, symptom_cols):
        symptoms_df = self.df[symptom_cols].copy().astype(bool).astype(int)
        return symptoms_df[symptoms_df.sum(axis=1) > 0]

    def _run_apriori_analysis(self, symptoms_df, params):
        frequent_itemsets = apriori(symptoms_df, min_support=params.min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return None, None, f"No frequent itemsets found with minimum support {params.min_support}."
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=params.min_confidence)
        strong_rules = rules[
            (rules['confidence'] > params.min_confidence) & 
            (rules['lift'] > params.min_lift)
        ].sort_values('confidence', ascending=False)
        
        return frequent_itemsets, strong_rules, None

    def _export_csv(self, strong_rules):
        csv_data = [{
            'source': list(rule['antecedents'])[0],
            'target': list(rule['consequents'])[0],
            'weight': rule['confidence'],
            'support': rule['support'],
            'lift': rule['lift'],
            'label': f"{rule['confidence']:.0%}"
        } for _, rule in strong_rules.iterrows()]
        pd.DataFrame(csv_data).to_csv('association_rules.csv', index=False)

    def _build_result_structure(self, params, symptom_cols, observation_names, 
                               condition_names, frequent_itemsets, strong_rules, symptoms_df):
        result = {
            "parameters": {
                "min_support": params.min_support,
                "min_confidence": params.min_confidence,
                "min_lift": params.min_lift,
                "total_symptoms_analyzed": len(symptom_cols),
                "total_patients_with_symptoms": len(symptoms_df)
            },
            "symptom_columns": symptom_cols,
            "observation_columns": [col for col in symptom_cols if col in observation_names],
            "condition_columns": [col for col in symptom_cols if col in condition_names],
            "frequent_itemsets": {
                "count": len(frequent_itemsets),
                "itemsets": [{
                    "itemset": list(row['itemsets']),
                    "support": float(row['support'])
                } for _, row in frequent_itemsets.iterrows()]
            },
            "association_rules": {
                "total_rules": len(strong_rules) if not strong_rules.empty else 0,
                "strong_rules_count": len(strong_rules),
                "rules": []
            }
        }

        for _, rule in strong_rules.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            conviction = rule.get('conviction')
            
            result["association_rules"]["rules"].append({
                "antecedents": antecedents,
                "consequents": consequents,
                "antecedent_observations": [i for i in antecedents if i in observation_names],
                "antecedent_conditions": [i for i in antecedents if i in condition_names],
                "consequent_observations": [i for i in consequents if i in observation_names],
                "consequent_conditions": [i for i in consequents if i in condition_names],
                "support": float(rule['support']),
                "confidence": float(rule['confidence']),
                "lift": float(rule['lift']),
                "conviction": float(conviction) if pd.notna(conviction) and np.isfinite(conviction) else None
            })
        
        return result

    def _add_network_analysis(self, result, strong_rules, observation_names, condition_names):
        G = nx.DiGraph()
        for _, rule in strong_rules.iterrows():
            ant = list(rule['antecedents'])[0]
            con = list(rule['consequents'])[0]
            G.add_edge(ant, con, weight=float(rule['confidence']), 
                      support=float(rule['support']), lift=float(rule['lift']),
                      label=f"{rule['confidence']:.0%}")

        centrality_metrics = {
            'degree': nx.degree_centrality(G),
            'in_degree': nx.in_degree_centrality(G),
            'out_degree': nx.out_degree_centrality(G)
        }

        node_data = []
        for node in G.nodes():
            node_type = ("observation" if node in observation_names else 
                        "condition" if node in condition_names else "unknown")
            node_data.append({
                "node": node,
                "type": node_type,
                "degree_centrality": centrality_metrics['degree'].get(node, 0),
                "in_degree_centrality": centrality_metrics['in_degree'].get(node, 0),
                "out_degree_centrality": centrality_metrics['out_degree'].get(node, 0)
            })

        result["network_analysis"] = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "nodes": sorted(node_data, key=lambda x: x["degree_centrality"], reverse=True),
            "most_central_nodes": sorted(node_data, key=lambda x: x["degree_centrality"], reverse=True)[:10]
        }

    def symptom_pattern_analysis_analysis(self, obs_names_path='obs_names.pkl', 
                                        cond_names_path='cond_names.pkl',
                                        min_support=0.1, min_confidence=0.7, min_lift=1.2,
                                        exclude_cols=None, include_network_analysis=True, 
                                        export_csv=False) -> dict:
        try:
            params = SymptomPatternParams(
                obs_names_path=obs_names_path, cond_names_path=cond_names_path,
                min_support=min_support, min_confidence=min_confidence, min_lift=min_lift,
                exclude_cols=exclude_cols or [], include_network_analysis=include_network_analysis,
                export_csv=export_csv
            )

            observation_names, condition_names = self._load_mappings(
                params.obs_names_path, params.cond_names_path
            )
            
            symptom_cols = self._identify_symptom_columns(params, observation_names, condition_names)
            if not symptom_cols:
                return {"error": "No suitable symptom/condition columns found for market basket analysis."}

            symptoms_df = self._prepare_symptoms_dataframe(symptom_cols)
            if symptoms_df.empty:
                return {"error": "No valid symptom patterns found in the data."}

            frequent_itemsets, strong_rules, error = self._run_apriori_analysis(symptoms_df, params)
            if error:
                return {"error": error}

            if params.export_csv and not strong_rules.empty:
                self._export_csv(strong_rules)

            result = self._build_result_structure(
                params, symptom_cols, observation_names, condition_names,
                frequent_itemsets, strong_rules, symptoms_df
            )

            if params.include_network_analysis and not strong_rules.empty:
                self._add_network_analysis(result, strong_rules, observation_names, condition_names)

            return result

        except Exception as e:
            return {"error": f"Error in market basket analysis: {str(e)}"}