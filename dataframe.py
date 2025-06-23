import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import click
# from icmr_viz.patient_data import patients_df

class ObservationRepository:
    def __init__(self, endpoint, page_size=50):
        self.endpoint = endpoint
        self.page_size = page_size

    def get_observations(self, patient_id):
        page_num = 1
        observations = []
        while True:
            observation_url = f"{self.endpoint}?patient={patient_id}&_count={self.page_size}&_page={page_num}"
            response = requests.get(observation_url)
            
            if response.status_code == 200:
                page_data = response.json()
                entries = page_data if isinstance(page_data, list) else page_data.get('entry', [])
                if not entries:
                    break  # No more entries, stop pagination
                observations.extend(entries)
                
                # Check if there's a next page
                next_link = None
                if isinstance(page_data, dict):
                    next_link = next((link['url'] for link in page_data.get('link', []) if link['relation'] == 'next'), None)
                if not next_link:
                    break  
                page_num += 1
            else:
                break  
        
        return observations



# %%
 # Define the ConditionRepository class
class ConditionRepository:
    def __init__(self, endpoint, page_size=50):
        self.endpoint = endpoint
        self.page_size = page_size

    def get_conditions(self, patient_id):
        page_num = 1
        conditions = []
        while True:
            condition_url = f"{self.endpoint}?patient={patient_id}&_count={self.page_size}&_page={page_num}"
            response = requests.get(condition_url)
            
            if response.status_code == 200:
                page_data = response.json()
                entries = page_data if isinstance(page_data, list) else page_data.get('entry', [])
                if not entries:
                    break  # No more entries, stop pagination
                conditions.extend(entries)
                
                # Check if there's a next page
                next_link = None
                if isinstance(page_data, dict):
                    next_link = next((link['url'] for link in page_data.get('link', []) if link['relation'] == 'next'), None)
                if not next_link:
                    break  # No more pages, stop pagination
                page_num += 1
            else:
                break  # Error response, stop pagination
        
        return conditions

# %%
import pandas.api.extensions
import re

# Registering a custom accessor for pandas Series
@pd.api.extensions.register_series_accessor("snomed_cts")
class SnomedAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not isinstance(obj, pd.Series):
            raise AttributeError("Must be a pandas Series")

    @property
    def metadata(self):
        df = self._obj._dataframe
        column_name = self._obj.name
        if hasattr(df, '_snomed_cts_metadata'):
            return df._snomed_cts_metadata.get(column_name, None)
        return None
    
    @property
    def display_name(self):
        metadata = self.metadata
        if metadata:
            return metadata.get('display_name', self._obj.name)
        return self._obj.name
    
    @property
    def code(self):
        metadata = self.metadata
        if metadata:
            return metadata.get('code', None)
        return None

# Custom DataFrame class to store additional metadata
class CustomDataFrame(pd.DataFrame):
    _metadata = ['_snomed_cts_metadata']

    def __init__(self, *args, **kwargs):
        self._snomed_cts_metadata = {}
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return CustomDataFrame

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, pd.Series):
            result._dataframe = self
        return result

    def set_snomed_cts(self, column_name, metadata):
        self._snomed_cts_metadata[column_name] = metadata

    def get_snomed_cts(self, column_name):
        return self._snomed_cts_metadata.get(column_name, None)

# Class for processing patient data
class PatientDataProcessor:
    def __init__(self, observation_repo, condition_repo, patients_df):
        self.observation_repo = observation_repo
        self.condition_repo = condition_repo
        self.patients_df = patients_df
        self.observation_names = []
        self.condition_names = []

    def get_patient_data(self, patient_id):
        observations = self.observation_repo.get_observations(patient_id)
        conditions = self.condition_repo.get_conditions(patient_id)
        
        patient_data = {'patient_id': patient_id}
        
        # Extract observation data
        for observation in observations:
            resource = observation['resource']
            code_display_pairs = []
            if 'coding' in resource.get('code', {}):
                for coding in resource['code']['coding']:
                    display = coding.get('display')
                    if display:
                        code_display_pairs.append(display)
            if code_display_pairs:
                value = resource.get('valueInteger', None)  
                if value is not None:  
                    observation_value = " | ".join(code_display_pairs)  
                    patient_data[observation_value] = value  # Store the value with concatenated pairs
                    for display in code_display_pairs:
                        if display not in self.observation_names:
                            self.observation_names.append(display)

        # Extract condition data
        for condition in conditions:
            resource = condition['resource']
            code = resource.get('verificationStatus', {}).get('coding', [{}])[0].get('code', None)
            if code == 'confirmed':
                condition_name = resource.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown')
                patient_data[condition_name] = 1
                if condition_name not in self.condition_names:
                    self.condition_names.append(condition_name)
        
        return patient_data
        
    def process_patient_data(self):
        # Create a DataFrame to store patient data with all observations and conditions
        df_patient_data_all = pd.DataFrame(columns=['patient_id'])
        # Populate DataFrame with patient IDs
        df_patient_data_all['patient_id'] = self.patients_df['patient_id']
        # Extract all observation and condition data for each patient
        patient_data_list = [self.get_patient_data(patient_id) for patient_id in df_patient_data_all['patient_id']]
        # Convert list of dictionaries into DataFrame
        df_patient_data_with_all_features = pd.DataFrame(patient_data_list).fillna(0)
        # Merge patient information DataFrame with DataFrame containing all observations and conditions
        df_patient_data_with_all_features = pd.merge(self.patients_df, df_patient_data_with_all_features, on='patient_id')

        # Convert to CustomDataFrame and set SNOMED CT metadata
        df_patient_data_with_all_features = CustomDataFrame(df_patient_data_with_all_features)
        
        for display_name in self.observation_names:
            snomed_cts = self.get_snomed_cts(display_name)
            metadata = snomed_cts if snomed_cts else {}
            metadata['display_name'] = display_name
            df_patient_data_with_all_features.set_snomed_cts(display_name, metadata)
        
        for display_name in self.condition_names:
            snomed_cts = self.get_snomed_cts(display_name)
            metadata = snomed_cts if snomed_cts else {}
            metadata['display_name'] = display_name
            df_patient_data_with_all_features.set_snomed_cts(display_name, metadata)
        
        return df_patient_data_with_all_features

    def get_observation_names(self):
        return self.observation_names
    
    def get_condition_names(self):
        return self.condition_names

    def get_snomed_cts(self, column_name):
        # For demonstration purposes, fetch metadata for the first patient
        patient_id = self.patients_df['patient_id'].iloc[0]
        
        # Check observations
        if column_name in self.observation_names:
            observations = self.observation_repo.get_observations(patient_id)
            for observation in observations:
                resource = observation['resource']
                if 'coding' in resource.get('code', {}):
                    for coding in resource['code']['coding']:
                        if coding.get('display') == column_name:
                            return {
                                "display_name": coding.get("display", None),
                                "code": coding.get("code", None),
                                "system": coding.get("system", None)
                            }
        # Check conditions
        if column_name in self.condition_names:
            conditions = self.condition_repo.get_conditions(patient_id)
            for condition in conditions:
                resource = condition['resource']
                if 'coding' in resource.get('code', {}):
                    for coding in resource['code']['coding']:
                        if coding.get('display') == column_name:
                            return {
                                "display_name": coding.get("display", None),
                                "code": coding.get("code", None),
                                "system": coding.get("system", None)
                            }
        return None