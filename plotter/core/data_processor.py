import pandas as pd
import numpy as np
import os

def prepare_data(data):
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        if 'cluster_data' in data:
            cluster_df = pd.DataFrame(data['cluster_data'])
            return cluster_df
        
        elif 'top_clusters' in data:
            cluster_data = []
            
            all_features = set()
            for cluster_id, info in data['top_clusters'].items():
                if 'means' in info:
                    all_features.update(info['means'].keys())
            
            all_features = sorted(list(all_features))
            
            for cluster_id, info in data['top_clusters'].items():
                cluster_size = info.get('size', 1)
                cluster_means = info.get('means', {})
                
                num_points = max(1, min(cluster_size, 50))
                
                for i in range(num_points):
                    data_point = {
                        'Cluster': f"Cluster {cluster_id}",
                        'cluster_id': int(cluster_id),
                        'point_id': i
                    }
                    
                    for feature in all_features:
                        mean_val = cluster_means.get(feature, 0)
                        variation = max(abs(mean_val) * 0.1, 0.1)
                        data_point[feature] = mean_val + np.random.normal(0, variation/3)
                    
                    cluster_data.append(data_point)
            
            return pd.DataFrame(cluster_data)
        
        elif 'stds' in data:
            items = list(data['stds'].items())
            items.sort(key=lambda x: x[1], reverse=True)
            return pd.DataFrame(items, columns=['Feature', 'Standard_Deviation'])
        
        elif 'means' in data:
            items = list(data['means'].items())
            items.sort(key=lambda x: abs(x[1]), reverse=True)
            return pd.DataFrame(items, columns=['Feature', 'Mean'])
        
        elif 'medians' in data:
            items = list(data['medians'].items())
            items.sort(key=lambda x: abs(x[1]), reverse=True)
            return pd.DataFrame(items, columns=['Feature', 'Median'])
        
        elif 'modes' in data:
            mode_data = []
            for col, mode_info in data['modes'].items():
                if isinstance(mode_info, dict) and 'mode' in mode_info:
                    mode_val = mode_info['mode']
                else:
                    mode_val = mode_info
                
                if isinstance(mode_val, list):
                    mode_str = ', '.join(map(str, mode_val))
                    mode_data.append({'Feature': col, 'Mode': mode_str, 'Multiple_Modes': True})
                else:
                    mode_data.append({'Feature': col, 'Mode': str(mode_val), 'Multiple_Modes': False})
            return pd.DataFrame(mode_data)
        
        elif 'ranges' in data:
            range_data = []
            for col, range_info in data['ranges'].items():
                range_data.append({
                    'Feature': col, 
                    'Range': range_info['range'],
                    'Min': range_info['min'],
                    'Max': range_info['max']
                })
            range_data.sort(key=lambda x: x['Range'], reverse=True)
            return pd.DataFrame(range_data)
        
        elif 'frequencies' in data:
            freq_data = []
            for col, freq_dict in data['frequencies'].items():
                for category, count in freq_dict.items():
                    freq_data.append({'Column': col, 'Category': category, 'Count': count})
            return pd.DataFrame(freq_data)
        
        elif 'frequency' in data:
            freq_data = []
            for category, count in data['frequency'].items():
                freq_data.append({'Category': category, 'Count': count})
            freq_data.sort(key=lambda x: x['Count'], reverse=True)
            return pd.DataFrame(freq_data)
        
        elif 'std' in data:
            return pd.DataFrame([{'Operation': 'Standard Deviation', 'Value': data['std']}])
        elif 'mean' in data:
            return pd.DataFrame([{'Operation': 'Mean', 'Value': data['mean']}])
        elif 'median' in data:
            return pd.DataFrame([{'Operation': 'Median', 'Value': data['median']}])
        elif 'mode' in data:
            mode_val = data['mode']
            if isinstance(mode_val, list):
                mode_str = ', '.join(map(str, mode_val))
            else:
                mode_str = str(mode_val)
            return pd.DataFrame([{'Operation': 'Mode', 'Value': mode_str}])
        elif 'range' in data:
            return pd.DataFrame([{
                'Operation': 'Range', 
                'Value': data['range'],
                'Min': data.get('min', ''),
                'Max': data.get('max', '')
            }])
        
        else:
            items = list(data.items())
            try:
                items.sort(key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
            except:
                pass
            return pd.DataFrame(items, columns=['Category', 'Value'])
    else:
        raise ValueError("Data must be a pandas DataFrame or dictionary")

def convert_operation_data_to_df(data, operation_type, top_n=15):
    
    if operation_type == 'mean':
        if 'means' in data:
            means_data = data['means']
            sorted_means = sorted(means_data.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            df = pd.DataFrame(sorted_means, columns=['Feature', 'Mean'])
            return df
    
    elif operation_type == 'std':
        if 'stds' in data:
            stds_data = data['stds']
            sorted_stds = sorted(stds_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_stds, columns=['Feature', 'Standard_Deviation'])
            return df
    
    elif operation_type == 'median':
        if 'medians' in data:
            medians_data = data['medians']
            sorted_medians = sorted(medians_data.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            df = pd.DataFrame(sorted_medians, columns=['Feature', 'Median'])
            return df
    
    elif operation_type == 'mode':
        if 'modes' in data:
            mode_data = []
            for col, mode_info in data['modes'].items():
                if isinstance(mode_info, dict) and 'mode' in mode_info:
                    mode_val = mode_info['mode']
                else:
                    mode_val = mode_info
                
                if isinstance(mode_val, list):
                    mode_str = ', '.join(map(str, mode_val))
                    mode_data.append({'Feature': col, 'Mode': mode_str})
                else:
                    mode_data.append({'Feature': col, 'Mode': str(mode_val)})
            df = pd.DataFrame(mode_data[:top_n])
            return df
    
    elif operation_type == 'range':
        if 'ranges' in data:
            ranges_data = data['ranges']
            range_items = [(k, v['range'], v['min'], v['max']) for k, v in ranges_data.items()]
            sorted_ranges = sorted(range_items, key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_ranges, columns=['Feature', 'Range', 'Min', 'Max'])
            return df
    
    elif operation_type == 'frequency':
        if 'frequencies' in data:
            all_freq_data = []
            for col, freq_dict in data['frequencies'].items():
                for category, count in freq_dict.items():
                    all_freq_data.append({'Column': col, 'Category': category, 'Frequency': count})
            df = pd.DataFrame(all_freq_data[:top_n])
            return df
        elif 'frequency' in data:
            freq_data = data['frequency']
            sorted_freq = sorted(freq_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_freq, columns=['Category', 'Frequency'])
            return df
    
    elif operation_type == 'clustering':
        if 'cluster_data' in data:
            cluster_df = pd.DataFrame(data['cluster_data'])
            return cluster_df
        elif 'top_clusters' in data:
            cluster_data = []
            
            all_features = set()
            for cluster_id, info in data['top_clusters'].items():
                if 'means' in info:
                    all_features.update(info['means'].keys())
            
            all_features = sorted(list(all_features))
            
            for cluster_id, info in data['top_clusters'].items():
                cluster_size = info.get('size', 1)
                cluster_means = info.get('means', {})
                
                num_points = max(1, min(cluster_size, 50))
                
                for i in range(num_points):
                    data_point = {
                        'Cluster': f"Cluster {cluster_id}",
                        'cluster_id': int(cluster_id),
                        'point_id': i
                    }
                    
                    for feature in all_features:
                        mean_val = cluster_means.get(feature, 0)
                        variation = max(abs(mean_val) * 0.1, 0.1)
                        data_point[feature] = mean_val + np.random.normal(0, variation/3)
                    
                    cluster_data.append(data_point)
            
            df = pd.DataFrame(cluster_data)
            return df

    elif operation_type == 'symptom-pattern':
        csv_filename = 'association_rules.csv'
        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
            return df.head(top_n)
        
        if 'association_rules' in data and 'rules' in data['association_rules']:
            rules = data['association_rules']['rules']
            if rules:
                network_data = []
                for rule in rules[:top_n]:
                    antecedent = rule['antecedents'][0] if isinstance(rule['antecedents'], list) else rule['antecedents']
                    consequent = rule['consequents'][0] if isinstance(rule['consequents'], list) else rule['consequents']
                    
                    network_data.append({
                        'source': antecedent,
                        'target': consequent,
                        'weight': rule['confidence'],
                        'support': rule['support'],
                        'lift': rule['lift'],
                        'label': f"{rule['confidence']:.0%}"
                    })
                
                df = pd.DataFrame(network_data)
                return df
    
    elif operation_type == 'prevalence':
        if 'results_by_column' in data:
            prev_data = []
            for col, prev_info in data['results_by_column'].items():
                prev_data.append({
                    'Column': col,
                    'Case_Value': prev_info.get('case_value', data.get('case_value', 'N/A')),
                    'N_Cases': prev_info.get('n_cases', 0),
                    'Total_Population': prev_info.get('total_population', 0),
                    'Prevalence_Proportion': prev_info.get('prevalence_proportion', 0.0),
                    'Prevalence_Percentage': prev_info.get('prevalence_percentage', 0.0)
                })
            
            # Sort by prevalence percentage in descending order
            prev_data.sort(key=lambda x: x['Prevalence_Percentage'], reverse=True)
            df = pd.DataFrame(prev_data[:top_n])
            return df
    
    elif operation_type == 'corr_coefficient' or operation_type == 'correlation':
        if 'correlations' in data:
            corr_data = []
            for corr_info in data['correlations']:
                # Handle both correlation types
                col1 = corr_info.get('col1', '')
                col2 = corr_info.get('col2', '')
                
                # Get Pearson correlation
                pearson_coef = corr_info.get('pearson_coefficient', np.nan)
                pearson_pval = corr_info.get('pearson_pvalue', np.nan)
                
                # Get Spearman correlation
                spearman_coef = corr_info.get('spearman_coefficient', np.nan)
                spearman_pval = corr_info.get('spearman_pvalue', np.nan)
                
                # Skip if both coefficients are NaN
                if not (pd.isna(pearson_coef) and pd.isna(spearman_coef)):
                    corr_data.append({
                        'Column_Pair': f"{col1} vs {col2}",
                        'Column1': col1,
                        'Column2': col2,
                        'Pearson_Coefficient': pearson_coef,
                        'Pearson_P_Value': pearson_pval,
                        'Spearman_Coefficient': spearman_coef,
                        'Spearman_P_Value': spearman_pval,
                        'Abs_Pearson': abs(pearson_coef) if not pd.isna(pearson_coef) else 0,
                        'Abs_Spearman': abs(spearman_coef) if not pd.isna(spearman_coef) else 0
                    })
            
            if corr_data:
                df = pd.DataFrame(corr_data)
                # Sort by absolute Pearson coefficient first, then by absolute Spearman
                df = df.sort_values(['Abs_Pearson', 'Abs_Spearman'], ascending=False)
                return df[:top_n]
    
    elif operation_type == 'covariance':
        if 'covariances' in data:
            cov_data = []
            for cov_pair, cov_info in data['covariances'].items():
                cov_data.append({
                    'Column_Pair': cov_pair,
                    'Column1': cov_info.get('column1', ''),
                    'Column2': cov_info.get('column2', ''),
                    'Covariance': cov_info.get('covariance', 0.0),
                    'Abs_Covariance': abs(cov_info.get('covariance', 0.0))
                })
            
            if cov_data:
                df = pd.DataFrame(cov_data)
                # Sort by absolute covariance value
                df = df.sort_values('Abs_Covariance', ascending=False)
                return df[:top_n]
    
    elif operation_type == 'observation':
        if isinstance(data, dict):
            sorted_obs = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_obs, columns=['Observation', 'Count'])
            return df
    
    elif operation_type == 'condition':
        if isinstance(data, dict):
            sorted_cond = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_cond, columns=['Condition', 'Count'])
            df['Condition_Clean'] = df['Condition'].apply(lambda x: x.replace('cond_', '').replace(' | ', ' + '))
            return df
    
    elif operation_type == 'correlation':
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            df = df.dropna(subset=['Correlation'])
            df = df.reindex(df['Correlation'].abs().sort_values(ascending=False).index)
            return df[:top_n]
    
    # Handle general dictionary data
    if isinstance(data, dict):
        if all(isinstance(v, (int, float)) for v in data.values()):
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            if operation_type in ['condition', 'observation']:
                col_names = [operation_type.capitalize(), 'Count']
            elif operation_type == 'frequency':
                col_names = ['Category', 'Frequency']
            else:
                col_names = ['Category', 'Value']
            
            df = pd.DataFrame(sorted_items, columns=col_names)
            return df
        
        # Handle nested dictionary structures
        for key in ['means', 'medians', 'modes', 'stds', 'ranges']:
            if key in data:
                items = list(data[key].items())[:top_n]
                df = pd.DataFrame(items, columns=['Feature', key.capitalize().rstrip('s')])
                return df
    
    print(f"Warning: Could not parse data for operation type '{operation_type}'")
    print(f"Data type: {type(data)}")
    print(f"Data sample: {str(data)[:200]}...")
    
    # Return empty DataFrame if nothing else worked
    return pd.DataFrame()

def get_y_label_for_operation(operation_type):
    labels = {
        'mean': 'Mean Value',
        'median': 'Median Value',
        'mode': 'Mode Value',
        'std': 'Standard Deviation',
        'range': 'Range',
        'frequency': 'Frequency/Count',
        'clustering': 'Feature Value',
        'observation': 'Count',
        'condition': 'Count',
        'correlation': 'Correlation Value',
        'corr_coefficient': 'Correlation Coefficient',
        'covariance': 'Covariance Value',
        'prevalence': 'Prevalence Percentage',
        'symptom_pattern': 'Confidence/Support',
    }
    return labels.get(operation_type, 'Value')