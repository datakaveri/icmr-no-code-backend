import pandas as pd
import os
from typing import List, Dict, Optional, Tuple
import click

class DatasetJoiner:
    
    SUPPORTED_JOINS = [
        'inner', 'outer', 'left', 'right', 'cross'
    ]
    
    def __init__(self):
        self.datasets = {}
        self.patients_data = {}
    
    def load_dataset_by_filename(self, processed_file: str, patients_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data file not found: {processed_file}")
        if not os.path.exists(patients_file):
            raise FileNotFoundError(f"Patients data file not found: {patients_file}")
        
        processed_df = pd.read_csv(processed_file)
        patients_df = pd.read_csv(patients_file)
        
        return processed_df, patients_df
    
    def validate_join_columns(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                            join_columns: List[str]) -> bool:
        for col in join_columns:
            if col not in df1.columns:
                raise ValueError(f"Column '{col}' not found in first dataset")
            if col not in df2.columns:
                raise ValueError(f"Column '{col}' not found in second dataset")
        return True
    
    def perform_join(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                join_type: str, join_columns: Optional[List[str]] = None,
                suffixes: Tuple[str, str] = ('_x', '_y')) -> pd.DataFrame:
        
        if join_type not in self.SUPPORTED_JOINS:
            raise ValueError(f"Unsupported join type: {join_type}. "
                           f"Supported types: {', '.join(self.SUPPORTED_JOINS)}")
        
        if join_type == 'cross':
            df1['_cross_key'] = 1
            df2['_cross_key'] = 1
            result = pd.merge(df1, df2, on='_cross_key', suffixes=suffixes)
            return result.drop('_cross_key', axis=1)
        
        if not join_columns:
            if 'patient_id' in df1.columns and 'patient_id' in df2.columns:
                join_columns = ['patient_id']
            else:
                raise ValueError("Join columns must be specified for non-cross joins")
        
        self.validate_join_columns(df1, df2, join_columns)
        
        cols1 = set(df1.columns) - set(join_columns)
        cols2 = set(df2.columns) - set(join_columns)
        overlapping_cols = list(cols1 & cols2)
        
        result = pd.merge(
            df1,
            df2,
            how=join_type,
            on=join_columns,
            suffixes=suffixes
        )
        
        return result
    
    def join_multiple_datasets_by_files(self, file_pairs: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(file_pairs) < 2:
            raise ValueError("At least 2 datasets required for join operation")
        
        first_config = file_pairs[0]
        processed_result, patients_result = self.load_dataset_by_filename(
            first_config['processed_file'], 
            first_config['patients_file']
        )
        
        # Join remaining datasets
        for i, config in enumerate(file_pairs[1:], 1):
            processed_df, patients_df = self.load_dataset_by_filename(
                config['processed_file'], 
                config['patients_file']
            )
            
            # Get join configuration
            join_type = config.get('join_type', 'inner')
            join_columns = config.get('join_columns', ['patient_id'])
            suffixes = config.get('suffixes', (f'_ds{i}', f'_ds{i+1}'))
            
            # Join processed data
            processed_result = self.perform_join(
                processed_result, 
                processed_df,
                join_type=join_type,
                join_columns=join_columns,
                suffixes=suffixes
            )
            
            # Join patients data
            patients_result = self.perform_join(
                patients_result,
                patients_df,
                join_type=join_type,
                join_columns=join_columns,
                suffixes=suffixes
            )
        
        return processed_result, patients_result
    
    def save_joined_data(self, processed_df: pd.DataFrame, patients_df: pd.DataFrame,
                        output_prefix: str = "joined"):
        
        processed_output = f"{output_prefix}_processed_data.csv"
        patients_output = f"{output_prefix}_patients_df.csv"
        
        processed_df.to_csv(processed_output, index=False)
        patients_df.to_csv(patients_output, index=False)
        
        return processed_output, patients_output
    
    @staticmethod
    def get_available_csv_files() -> Dict[str, List[str]]:
        processed_files = []
        patients_files = []
        
        for file in os.listdir('.'):
            if file.endswith('_processed_data.csv') or file == 'processed_data.csv':
                processed_files.append(file)
            elif file.endswith('_patients_df.csv') or file == 'patients_df.csv':
                patients_files.append(file)
        
        return {
            'processed_files': sorted(processed_files),
            'patients_files': sorted(patients_files)
        }
    
    def preview_join(self, dataset_configs: List[Dict], num_rows: int = 5) -> Dict:
        
        preview_configs = []
        for config in dataset_configs:
            processed_df, patients_df = self.load_dataset(config['dataset'])
            
            processed_sample = processed_df.head(num_rows)
            patients_sample = patients_df.head(num_rows)
            
            preview_configs.append({
                'dataset': config['dataset'],
                'processed_shape': processed_df.shape,
                'patients_shape': patients_df.shape,
                'processed_columns': list(processed_df.columns),
                'patients_columns': list(patients_df.columns),
                'processed_sample': processed_sample,
                'patients_sample': patients_sample
            })
        
        return {
            'datasets': preview_configs,
            'estimated_result_size': 'Depends on join type and data overlap'
        }