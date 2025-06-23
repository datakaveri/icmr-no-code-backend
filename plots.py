import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import click
import plotly.express as px
import plotly.io as pio
from icmr_viz.dataframe import patients_df, observation_repo, condition_repo, processed_data, observation_names, condition_names

observation_data = {}

def plot_observation_data():
    observation_data = {}
    for patient_id in patients_df['patient_id']:
        observations = observation_repo.get_observations(patient_id)
        for observation in observations:
            resource = observation['resource']
            code_display_pairs = []
            if 'coding' in resource.get('code', {}):
                for coding in resource['code']['coding']:
                    display = coding.get('display')
                    if display:
                        code_display_pairs.append(display)
            code_display_pairs = " | ".join(code_display_pairs)
            if code_display_pairs:
                value = resource.get('valueBoolean', None)
                if value is not None and value:  
                    observation_data[code_display_pairs] = observation_data.get(code_display_pairs, 0) + 1 if value else 0
    
    plt.figure(figsize=(10, 6))
    plt.bar(observation_data.keys(), observation_data.values(), color='blue')
    plt.title('Count of True Values for Each Observation Code')
    plt.xlabel('Observation Code')
    plt.ylabel('Count of True Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

plot_observation_data()

# %%
condition_data = {}

def plot_condition_data():
    condition_data = {}
    for patient_id in patients_df['patient_id']:
        conditions = condition_repo.get_conditions(patient_id)
        for condition in conditions:
            resource = condition['resource']
            code_display_pairs = []
            code = resource.get('verificationStatus', {}).get('coding', [{}])[0].get('code', None)
            if code == 'confirmed':
                if 'coding' in resource.get('code', {}):
                    for coding in resource['code']['coding']:
                        display = coding.get('display')
                        if display:
                            code_display_pairs.append(display)
            code_display_pairs = " | ".join(code_display_pairs)
            if code_display_pairs:
                condition_data[code_display_pairs] = condition_data.get(code_display_pairs, 0) + 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(condition_data.keys(), condition_data.values(), color='green')
    plt.title("Count of 'Confirmed' Conditions for Each Condition")
    plt.xlabel('Condition Name')
    plt.ylabel('Count of "Confirmed" Conditions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

plot_condition_data()

# %%

def plot_correlation():
    processed_data = pd.read_csv('processed_data.csv')
    
    df_corr = processed_data.drop(columns=["patient_id", "gender", "active", "last_updated"]).reset_index()
    constant_value_cols = df_corr.columns[df_corr.apply(pd.Series.nunique) == 1].tolist()

    for condition_name in condition_names:
        if condition_name in constant_value_cols:
            condition_names.remove(condition_name)
    for observation_name in observation_names:
        if observation_name in constant_value_cols:
            observation_names.remove(observation_name)

    df_corr = df_corr.loc[:, df_corr.apply(pd.Series.nunique) != 1]
    df_corr = df_corr.drop(columns=['index']).corr().reset_index()

    xs = ["index"] + condition_names
    df_corr_filtered = df_corr[df_corr["index"].isin(observation_names)][xs].set_index("index")
    
    fig = px.imshow(df_corr_filtered, 
                    labels=dict(x="Conditions", y="Observations", color="Correlation"),
                    x=condition_names, 
                    y=df_corr_filtered.index, 
                    color_continuous_scale='YlGnBu',
                    aspect="auto")
    
    fig.show()

plot_correlation()



