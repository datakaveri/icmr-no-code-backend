
import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import click
import pickle
import os
import mpld3
from fpdf import FPDF
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from icmr_viz.patient_data import FHIRData, PatientRepository
from icmr_viz.dataframe import PatientDataProcessor, CustomDataFrame, ObservationRepository, ConditionRepository
from operations import StatisticalOperations, save_results_to_json
from plotter import GenericPlotter, create_plot,convert_operation_data_to_df,get_y_label_for_operation
from join_operations import DatasetJoiner
import string
import base64
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)

def abbreviate_name(name, prefix, duplicate_col_check, duplicate_counter):
    vowels = "aeiouAEIOU0123456789"
    sct = "".join([char for char in name if char not in vowels])
    sct = sct.strip().replace(" ", "")
    sct = remove_punctuation(sct)
    abbr_col = sct[:3]  

    if abbr_col in duplicate_col_check:
        abbr_col = f"{abbr_col}{duplicate_counter}"
        duplicate_counter += 1
    return f"{prefix}_{abbr_col}", duplicate_counter

def save_plot_to_json(fig, plot_type, plot_name=None):
    """Save plot data to plotly.json file"""
    plotly_file = 'plotly.json'
    
    # Load existing data or create new structure
    if os.path.exists(plotly_file):
        with open(plotly_file, 'r') as f:
            plot_data = json.load(f)
    else:
        plot_data = {'plots': []}
    
    # Create plot entry
    plot_entry = {
        'type': plot_type,
        'data': pio.to_json(fig)
    }
    
    if plot_name:
        plot_entry['name'] = plot_name
    
    # Add to plots list
    plot_data['plots'].append(plot_entry)

    with open(plotly_file, 'w') as f:
        json.dump(plot_data, f, indent=2)

@click.command(help=click.style("Downloads the FHIR dataframe to patient_data.csv file by default (to store it in a custom location)", fg='yellow'))
@click.option('--base_url', default='http://localhost:8080/fhir', type=str, help='FHIR Base URL')
@click.option('--processed_data_path', '-o2', type=click.Path(exists=False, writable=True, dir_okay=False, resolve_path=True), default='processed_data.csv', help='Output Processed data CSV file name')
@click.option('--patients_df_path', '-o1', type=click.Path(exists=False, writable=True, dir_okay=False, resolve_path=True), default='patients_df.csv', help='Output Patients CSV file name')
@click.option('--obs_names_path', '-o3', type=click.Path(exists=False, writable=True, dir_okay=False, resolve_path=True), default='obs_names.pkl', help='Output Observation names file name')
@click.option('--cond_names_path', '-o4', type=click.Path(exists=False, writable=True, dir_okay=False, resolve_path=True), default='cond_names.pkl', help='Output Condition names file name')
@click.option('--dataset_name', '-d', required=True, type=str, help='Comma-separated list of dataset names to filter the data')
def download_data(base_url, processed_data_path, patients_df_path, dataset_name, obs_names_path, cond_names_path):
    datasets = dataset_name.split(',')
    
    config_data = {"base_url": base_url}
    with open("config.json", "w") as config_file:
        json.dump(config_data, config_file)

    all_patients_df = pd.DataFrame()
    all_processed_data = pd.DataFrame()
    all_observation_names = set()
    all_condition_names = set()

    for dataset in datasets:
        dataset = dataset.strip()
        if dataset:
            fhir = FHIRData(base_url, dataset)
            repository = PatientRepository(fhir)
            patients_df = repository.get_patients_dataframe()
            observation_repo = ObservationRepository(f"{base_url}/Observation")
            condition_repo = ConditionRepository(f"{base_url}/Condition")
            processor = PatientDataProcessor(observation_repo, condition_repo, patients_df)
            processed_data = processor.process_patient_data()
            observation_names = processor.observation_names
            condition_names = processor.condition_names
            
            all_patients_df = pd.concat([all_patients_df, patients_df])
            all_processed_data = pd.concat([all_processed_data, processed_data])
            all_observation_names.update(observation_names)
            all_condition_names.update(condition_names)
            
    all_patients_df.to_csv(patients_df_path, index=False)
    all_processed_data.to_csv(processed_data_path, index=False)

    with open(obs_names_path, 'wb') as f:
        pickle.dump(list(all_observation_names), f)
    
    with open(cond_names_path, 'wb') as f:
        pickle.dump(list(all_condition_names), f)
    
    plotly_data = {'plots': []}
    with open('plotly.json', 'w') as f:
        json.dump(plotly_data, f, indent=2)

    click.echo(f"Patients data saved to {patients_df_path}")
    click.echo(f"Processed data saved to {processed_data_path}")
    click.echo(f"Observation names saved to {obs_names_path}")
    click.echo(f"Condition names saved to {cond_names_path}")
    click.echo(f"Plotly JSON file initialized: plotly.json")
    click.echo(all_processed_data.columns)

    with open(obs_names_path, 'rb') as f:
        obs_names = pickle.load(f)
        click.echo(f"Contents of {obs_names_path}: {obs_names}")
    
    with open(cond_names_path, 'rb') as f:
        cond_names = pickle.load(f)
        click.echo(f"Contents of {cond_names_path}: {cond_names}")

@click.command(help=click.style("Join multiple datasets with various join operations", fg='cyan'))
@click.option('--processed_files', '-pf', required=True, type=str, help='Comma-separated list of processed data CSV files to join')
@click.option('--patients_files', '-df', required=True, type=str, help='Comma-separated list of patients data CSV files to join')
@click.option('--join_types', '-j', type=str, default='inner',help='Comma-separated list of join types (inner,outer,left,right,cross). Default: inner')
@click.option('--join_columns', '-c', type=str, default='patient_id', help='Comma-separated list of columns to join on. Default: patient_id')
@click.option('--output_processed', '-op', type=str, default='joined_processed_data.csv',help='Output processed data file name. Default: joined_processed_data.csv')
@click.option('--output_patients', '-od', type=str, default='joined_patients_df.csv', help='Output patients data file name. Default: joined_patients_df.csv')
@click.option('--preview', is_flag=True, default=False, help='Preview join operation without executing')
@click.option('--suffixes', '-s', type=str, default='_x,_y', help='Comma-separated suffixes for overlapping columns. Default: _x,_y')
def join(processed_files, patients_files, join_types, join_columns, 
                 output_processed, output_patients, preview, suffixes):
    
    try:
        processed_file_list = [f.strip() for f in processed_files.split(',')]
        patients_file_list = [f.strip() for f in patients_files.split(',')]
        join_type_list = [j.strip() for j in join_types.split(',')]
        join_column_list = [c.strip() for c in join_columns.split(',') if c.strip()]
        suffix_list = [s.strip() for s in suffixes.split(',')]
        
        if len(processed_file_list) != len(patients_file_list):
            raise click.ClickException("Number of processed files must match number of patients files")
        
        if len(processed_file_list) < 2:
            raise click.ClickException("At least 2 file pairs required for join operation")
        
        while len(join_type_list) < len(processed_file_list) - 1:
            join_type_list.append('inner')
        
        while len(suffix_list) < 2:
            suffix_list.extend(['_x', '_y'])
        
        joiner = DatasetJoiner()
        
        missing_files = []
        for pf, df in zip(processed_file_list, patients_file_list):
            if not os.path.exists(pf):
                missing_files.append(pf)
            if not os.path.exists(df):
                missing_files.append(df)
        
        if missing_files:
            available_files = joiner.get_available_csv_files()
            click.echo(click.style("Available processed files:", fg='yellow'))
            for pf in available_files['processed_files']:
                click.echo(f"  - {pf}")
            click.echo(click.style("Available patients files:", fg='yellow'))
            for df in available_files['patients_files']:
                click.echo(f"  - {df}")
            raise click.ClickException(f"Missing files: {', '.join(missing_files)}")
        
        file_pairs = []
        
        # First dataset (base)
        file_pairs.append({
            'processed_file': processed_file_list[0],
            'patients_file': patients_file_list[0],
            'dataset_label': f"dataset_1_{processed_file_list[0].split('.')[0]}",
            'join_type': None,  
            'join_columns': join_column_list if join_column_list else None,
            'suffixes': (suffix_list[0], suffix_list[1])
        })
        
        for i, (pf, df) in enumerate(zip(processed_file_list[1:], patients_file_list[1:]), 1):
            join_type = join_type_list[i-1] if i-1 < len(join_type_list) else 'inner'
            
            file_pairs.append({
                'processed_file': pf,
                'patients_file': df,
                'dataset_label': f"dataset_{i+1}_{pf.split('.')[0]}",
                'join_type': join_type,
                'join_columns': join_column_list if join_column_list else None,
                'suffixes': (f'_ds1', f'_ds{i+1}')
            })
        
        if preview:
            click.echo(click.style("Preview Mode - Join Configuration:", fg='cyan'))
            
            for i, config in enumerate(file_pairs):
                processed_df, patients_df = joiner.load_dataset_by_filename(
                    config['processed_file'], config['patients_file']
                )
                
                click.echo(f"\n{i+1}. Dataset: {config['dataset_label']}")
                click.echo(f"   Processed file: {config['processed_file']} (shape: {processed_df.shape})")
                click.echo(f"   Patients file: {config['patients_file']} (shape: {patients_df.shape})")
                click.echo(f"   Processed columns: {len(processed_df.columns)}")
                click.echo(f"   Patients columns: {len(patients_df.columns)}")
            
            click.echo(f"\nJoin Operations:")
            for i, config in enumerate(file_pairs[1:], 1):
                click.echo(f"  {i}. {config['join_type']} join on {config['join_columns']}")
            
            click.echo(click.style(f"\nOutput files would be:", fg='green'))
            click.echo(f"  - {output_processed}")
            click.echo(f"  - {output_patients}")
            click.echo(click.style("\nRun without --preview flag to execute the join.", fg='green'))
            return
        
        click.echo(click.style("Executing join operation...", fg='yellow'))
        
        processed_result, patients_result = joiner.join_multiple_datasets_by_files(file_pairs)
        
        processed_result.to_csv(output_processed, index=False)
        patients_result.to_csv(output_patients, index=False)
        
        click.echo(click.style("✓ Join operation completed successfully!", fg='green'))
        click.echo(f"  Joined processed data: {output_processed} ({len(processed_result)} records)")
        click.echo(f"  Joined patients data: {output_patients} ({len(patients_result)} records)")
        click.echo(f"  Total columns in processed data: {len(processed_result.columns)}")
        
        click.echo(f"\nJoin Summary:")
        for config in file_pairs[1:]:
            click.echo(f"  - {config['join_type']} join with {config['processed_file']}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error during join operation: {str(e)}", fg='red'))
        raise click.ClickException(f"Join operation failed: {str(e)}")


@click.command(help="Abbreviates column names in a processed_data CSV file and generates a CSV with original and abbreviated names.")
@click.option('--processed_data_path', '-i', type=click.Path(exists=True, dir_okay=False, resolve_path=True), default='processed_data.csv', required=True, help='Input Processed data CSV file name')
@click.option('--obs_names_path', '-o3', type=click.Path(exists=True, dir_okay=False, resolve_path=True), default='obs_names.pkl', required=True, help='Input Observation names file name')
@click.option('--cond_names_path', '-o4', type=click.Path(exists=True, dir_okay=False, resolve_path=True), default='cond_names.pkl', required=True, help='Input Condition names file name')
@click.option('--abbr_path', '-a', type=click.Path(exists=False, writable=True, dir_okay=False, resolve_path=True), default='abbreviation_data.csv', required=True, help='Output Abbreviation mapping CSV file name')
def abbreviate(processed_data_path, obs_names_path, cond_names_path, abbr_path):
    with open(obs_names_path, 'rb') as f:
        observation_names = pickle.load(f)
    with open(cond_names_path, 'rb') as f:
        condition_names = pickle.load(f)

    processed_data = pd.read_csv(processed_data_path)
    
    abbreviation_mapping = []
    duplicate_col_check = []
    duplicate_counter = 1

    for name in observation_names:
        abbr_name, duplicate_counter = abbreviate_name(name, 'obs', duplicate_col_check, duplicate_counter)
        abbreviation_mapping.append({'Original Name': name, 'Abbreviated Name': abbr_name})
        duplicate_col_check.append(abbr_name)

    for name in condition_names:
        abbr_name, duplicate_counter = abbreviate_name(name, 'cond', duplicate_col_check, duplicate_counter)
        abbreviation_mapping.append({'Original Name': name, 'Abbreviated Name': abbr_name})
        duplicate_col_check.append(abbr_name)

    abbreviation_df = pd.DataFrame(abbreviation_mapping)
    abbreviation_df.to_csv(abbr_path, index=False)

@click.command()
@click.option('--file', '-f', default='processed_data.csv',required=True, help='CSV file path')
@click.option('--column', '-c', help='Specific column name (optional)')
def mean(file, column):
    try:
        df = pd.read_csv(file)
        stats = StatisticalOperations(df)
        result = stats.calculate_mean(column)
        
        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return
        
        if column:
            click.echo(f"Mean of '{column}': {result['mean']:.4f}")
        else:
            click.echo("Means of all numeric columns:")
            for col, mean_val in result['means'].items():
                click.echo(f"  {col}: {mean_val:.4f}")
        
        
        filename = save_results_to_json(result, 'mean')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")

@click.command()
@click.option('--file', '-f', default='processed_data.csv',required=True, help='CSV file path')
@click.option('--column', '-c', help='Specific column name (optional)')
def median(file, column):
    try:
        df = pd.read_csv(file)
        stats = StatisticalOperations(df)
        result = stats.calculate_median(column)
        
        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return
        
        if column:
            click.echo(f"Median of '{column}': {result['median']:.4f}")
        else:
            click.echo("Medians of all numeric columns:")
            for col, median_val in result['medians'].items():
                click.echo(f"  {col}: {median_val:.4f}")
        
        
        filename = save_results_to_json(result, 'median')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")

@click.command()
@click.option('--file', '-f', default='processed_data.csv',required=True, help='CSV file path')
@click.option('--column', '-c', help='Specific column name (optional)')
def mode(file, column):
    """Calculate mode for numeric columns"""
    try:
        df = pd.read_csv(file)
        stats = StatisticalOperations(df)
        result = stats.calculate_mode(column)
        
        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return
        
        if column:
            mode_val = result['mode']
            if isinstance(mode_val, list):
                click.echo(f"Modes of '{column}': {mode_val} (multiple modes)")
            else:
                click.echo(f"Mode of '{column}': {mode_val}")
        else:
            click.echo("Modes of all numeric columns:")
            for col, mode_info in result['modes'].items():
                mode_val = mode_info['mode']
                if isinstance(mode_val, list):
                    click.echo(f"  {col}: {mode_val} (multiple modes)")
                else:
                    click.echo(f"  {col}: {mode_val}")
        
        
        filename = save_results_to_json(result, 'mode')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")

@click.command()
@click.option('--file', '-f',default='processed_data.csv',required=True, help='CSV file path')
@click.option('--column', '-c', help='Specific column name (optional)')
def std(file, column):
    try:
        df = pd.read_csv(file)
        stats = StatisticalOperations(df)
        result = stats.calculate_std(column)
        
        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return
        
        if column:
            click.echo(f"Standard deviation of '{column}': {result['std']:.4f}")
        else:
            click.echo("Standard deviations of all numeric columns:")
            for col, std_val in result['stds'].items():
                click.echo(f"  {col}: {std_val:.4f}")
        
        
        filename = save_results_to_json(result, 'std')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")

@click.command()
@click.option('--file', '-f',default='processed_data.csv' ,required=True, help='CSV file path')
@click.option('--column', '-c', help='Specific column name (optional)')
def range(file, column):
    try:
        df = pd.read_csv(file)
        stats = StatisticalOperations(df)
        result = stats.calculate_range(column)
        
        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return
        
        if column:
            click.echo(f"Range of '{column}': {result['range']:.4f} (min: {result['min']:.4f}, max: {result['max']:.4f})")
        else:
            click.echo("Ranges of all numeric columns:")
            for col, range_info in result['ranges'].items():
                click.echo(f"  {col}: {range_info['range']:.4f} (min: {range_info['min']:.4f}, max: {range_info['max']:.4f})")
        

        filename = save_results_to_json(result, 'range')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")

@click.command()
@click.option('--file', '-f',default='processed_data.csv' ,required=True, help='CSV file path')
@click.option('--column', '-c', help='Specific column name (optional)')
@click.option('--proportion', '-p', is_flag=True, help='Show proportions instead of counts')
def frequency(file, column, proportion):
    try:
        df = pd.read_csv(file)
        stats = StatisticalOperations(df)
        result = stats.frequency_analysis(column, proportion)
        
        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return
        
        if column:
            freq_type = "Proportions" if proportion else "Frequencies"
            click.echo(f"{freq_type} for '{column}':")
            for category, count in result['frequency'].items():
                if proportion:
                    click.echo(f"  {category}: {count:.4f}")
                else:
                    click.echo(f"  {category}: {count}")
        else:
            freq_type = "Proportions" if proportion else "Frequencies"
            click.echo(f"{freq_type} for all categorical columns:")
            for col, freq_dict in result['frequencies'].items():
                click.echo(f"  {col}:")
                for category, count in freq_dict.items():
                    if proportion:
                        click.echo(f"    {category}: {count:.4f}")
                    else:
                        click.echo(f"    {category}: {count}")
        
        
        filename = save_results_to_json(result, 'frequency')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")

@click.command()
@click.option('--file', '-f',default='processed_data.csv' ,required=True, help='CSV file path')
@click.option('--features', help='Comma-separated list of features to use for clustering')
@click.option('--clusters', '-k', default=3, help='Number of clusters (default: 3)')
@click.option('--topx', '-t', default=3, help='Number of top clusters to show (default: 3)')
def cluster(file, features, clusters, topx):
    try:
        df = pd.read_csv(file)
        stats = StatisticalOperations(df)
        result = stats.perform_clustering(features, clusters, topx)
        
        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return
        
        click.echo(f"Clustering Analysis with {result['clusters']} clusters:")
        click.echo(f"Features used: {', '.join(result['features'])}")
        click.echo(f"\nTop {topx} most distinct clusters:")
        
        for cluster_id, info in result['top_clusters'].items():
            click.echo(f"\nCluster {cluster_id}:")
            click.echo(f"  Size: {info['size']} samples")
            click.echo(f"  Distinctness Score: {info['distinctness']:.4f}")
            click.echo("  Feature Means:")
            for feature, mean_val in info['means'].items():
                click.echo(f"    {feature}: {mean_val:.4f}")
        
        
        filename = save_results_to_json(result, 'clustering')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True, readable=True), default='patients_df.csv', help='Provide the file name for Patients data')
def observation(input):
    try:
        # Load config
        with open("config.json", "r") as config_file:
            config_data = json.load(config_file)
            base_url = config_data.get("base_url")

        from icmr_viz.dataframe import ObservationRepository
        
        patients_df = pd.read_csv(input)
        observation_repo = ObservationRepository(f"{base_url}/Observation")
        observation_data = {}

        duplicate_col_check = []
        duplicate_counter = 1

        for patient_id in patients_df['patient_id']:
            observations = observation_repo.get_observations(patient_id)
            for observation in observations:
                resource = observation['resource']
                code_display_pairs = []
                
                if 'coding' in resource.get('code', {}):
                    for coding in resource['code']['coding']:
                        display = coding.get('display')
                        if display:
                            abbr_name, duplicate_counter = abbreviate_name(display, "obs", duplicate_col_check, duplicate_counter)
                            code_display_pairs.append(abbr_name)
                
                code_display_pairs = " | ".join(code_display_pairs)
                if code_display_pairs:
                    value = resource.get('valueBoolean', None)
                    if value is not None and value:
                        observation_data[code_display_pairs] = observation_data.get(code_display_pairs, 0) + 1

        result = observation_data  

        click.echo(f"Observation analysis completed:")
        click.echo(f"  Total observation types: {len(observation_data)}")
        click.echo(f"  Total true values: {sum(observation_data.values())}")
        
        if observation_data:
            click.echo("  Top 5 observations:")
            sorted_obs = sorted(observation_data.items(), key=lambda x: x[1], reverse=True)[:5]
            for obs, count in sorted_obs:
                click.echo(f"    {obs}: {count}")

        filename = save_results_to_json(result, 'observation')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")
        import traceback
        click.echo("Full error details:")
        click.echo(traceback.format_exc())

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True, readable=True), default='patients_df.csv', help='Provide the file name for Patients data')
def condition(input):
    """Calculate condition data counts for plotting"""
    try:
        # Load config
        with open("config.json", "r") as config_file:
            config_data = json.load(config_file)
            base_url = config_data.get("base_url")
        
        # Import required classes (assuming they're available in your environment)
        from icmr_viz.dataframe import ConditionRepository
        
        patients_df = pd.read_csv(input)
        condition_repo = ConditionRepository(f"{base_url}/Condition")
        condition_data = {}

        duplicate_col_check = []
        duplicate_counter = 1

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
                                abbr_name, duplicate_counter = abbreviate_name(display, "cond", duplicate_col_check, duplicate_counter)
                                code_display_pairs.append(abbr_name)
                
                code_display_pairs = " | ".join(code_display_pairs)
                if code_display_pairs:
                    condition_data[code_display_pairs] = condition_data.get(code_display_pairs, 0) + 1

        # Prepare result data for plotting - format it as the plotter expects
        result = condition_data  # Direct dictionary for plotting

        click.echo(f"Condition analysis completed:")
        click.echo(f"  Total condition types: {len(condition_data)}")
        click.echo(f"  Total confirmed conditions: {sum(condition_data.values())}")
        
        if condition_data:
            click.echo("  Top 5 conditions:")
            sorted_cond = sorted(condition_data.items(), key=lambda x: x[1], reverse=True)[:5]
            for cond, count in sorted_cond:
                click.echo(f"    {cond}: {count}")

        
        filename = save_results_to_json(result, 'condition')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")
        import traceback
        click.echo("Full error details:")
        click.echo(traceback.format_exc())

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True, readable=True), default='processed_data.csv', help='Provide the file name for processed data')
@click.option('--obs-names-path', type=click.Path(exists=True, readable=True), default='obs_names.pkl', help='Observation names pickle file')
@click.option('--cond-names-path', type=click.Path(exists=True, readable=True), default='cond_names.pkl', help='Condition names pickle file')
def correlation(input, obs_names_path, cond_names_path):
    try:
        processed_data = pd.read_csv(input)
        
        with open(obs_names_path, 'rb') as f:
            observation_names = pickle.load(f)
        with open(cond_names_path, 'rb') as f:
            condition_names = pickle.load(f)

        df_corr = processed_data.drop(columns=["patient_id", "gender", "active", "last_updated"]).reset_index()
        constant_value_cols = df_corr.columns[df_corr.apply(pd.Series.nunique) == 1].tolist()

        observation_names = [name for name in observation_names if name not in constant_value_cols]
        condition_names = [name for name in condition_names if name not in constant_value_cols]

        df_corr = df_corr.loc[:, df_corr.apply(pd.Series.nunique) != 1]
        df_corr = df_corr.drop(columns=['index']).corr().reset_index()

        xs = ["index"] + condition_names
        df_corr_filtered = df_corr[df_corr["index"].isin(observation_names)][xs].set_index("index")

        correlation_data = []
        for obs in df_corr_filtered.index:
            for cond in condition_names:
                if cond in df_corr_filtered.columns:
                    correlation_data.append({
                        'Observation': obs,
                        'Condition': cond,
                        'Correlation': df_corr_filtered.loc[obs, cond]
                    })

        correlation_df = pd.DataFrame(correlation_data)
        result = correlation_df.to_dict('records')  

        click.echo(f"Correlation analysis completed:")
        click.echo(f"  Matrix shape: {len(observation_names)} observations × {len(condition_names)} conditions")
        click.echo(f"  Total correlation pairs: {len(correlation_data)}")
        
        if correlation_data:
            correlations = [item['Correlation'] for item in correlation_data if not pd.isna(item['Correlation'])]
            if correlations:
                click.echo(f"  Correlation range: {min(correlations):.3f} to {max(correlations):.3f}")
                click.echo(f"  Average correlation: {np.mean(correlations):.3f}")


        filename = save_results_to_json(result, 'correlation')
        click.echo(f"Results saved to {filename}")
            
    except Exception as e:
        click.echo(f"Error: {e}")
        import traceback
        click.echo("Full error details:")
        click.echo(traceback.format_exc())

@click.command()
@click.option('--data-file', '-d', default='operations.json', help='JSON file containing operation results (default: operations.json)')
@click.option('--csv-file', '-f', default='processed_data.csv', help='CSV file to create plot from directly (default: processed_data.csv)')
@click.option('--plot-type', '-t', default='bar', type=click.Choice(['bar', 'line', 'scatter', 'histogram', 'box', 'violin', 'heatmap', 'pie']), help='Type of plot to create (default: bar)')
@click.option('--operation', '-op', help='Specific operation to plot (if multiple operations in JSON)')
@click.option('--title', help='Plot title (auto-generated if not provided)')
@click.option('--x-label', '-x', help='X-axis column name or label')
@click.option('--y-label', '-y', help='Y-axis column name or label')
@click.option('--color-column', '-c', help='Column name for color coding points/bars')
@click.option('--size-column', help='Column name for size coding (useful for scatter plots)')
@click.option('--facet-column', help='Column name for creating subplots/facets')
@click.option('--width', default=800, help='Plot width in pixels (default: 800)')
@click.option('--height', default=600, help='Plot height in pixels (default: 600)')
@click.option('--theme', default='plotly', type=click.Choice(['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white']), help='Plot theme/style (default: plotly)')
@click.option('--output', '-o', help='Output file path for PNG (auto-generated with timestamp if not provided)')
@click.option('--show', is_flag=True, help='Display the plot in browser after creation')
@click.option('--top-n', default=15, help='Show only top N features (default: 15)')
def plot(data_file, csv_file, plot_type, operation, title, x_label, y_label, color_column, size_column, facet_column, width, height, theme, output, show, top_n):
    """Create plots from operation results or CSV data"""
    try:
        data = None
        operation_type = None
        data_source = None
        
        # PRIORITY 1: Try to load from operations JSON first
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    json_data = json.load(f)
                
                if 'results' in json_data and len(json_data['results']) > 0:
                    # Find the specified operation or use the latest one
                    if operation:
                        # Find specific operation
                        target_result = None
                        for result in json_data['results']:
                            if result['operation'].lower() == operation.lower():
                                target_result = result
                                break
                        if target_result is None:
                            available_ops = [r['operation'] for r in json_data['results']]
                            click.echo(f"❌ Operation '{operation}' not found. Available operations: {available_ops}")
                            return
                    else:
                        # Use the latest operation
                        target_result = json_data['results'][-1]
                    
                    operation_type = target_result['operation']
                    raw_data = target_result['data']
                    data_source = f"operation '{operation_type}' from {data_file}"
                    
                    # Convert operation results to plottable format
                    data = convert_operation_data_to_df(raw_data, operation_type, top_n)
                    
                    if not data.empty:
                        if not title:
                            title = f"{operation_type.capitalize()} Analysis Results"
                        click.echo(f"📊 Successfully loaded {operation_type} operation data with {len(data)} entries")
                        click.echo(f"📋 Data columns: {list(data.columns)}")
                        
                        # Show sample of the data
                        if len(data) > 0:
                            click.echo("📈 Top 5 entries:")
                            for i, row in data.head(5).iterrows():
                                click.echo(f"   {row.iloc[0]}: {row.iloc[1]}")
                    else:
                        click.echo(f"⚠️  No plottable data found in {operation_type} operation results")
                        data = None
                else:
                    click.echo(f"⚠️  No results found in {data_file}")
            except Exception as e:
                click.echo(f"⚠️  Error reading {data_file}: {e}")
        
        # PRIORITY 2: Only fallback to CSV if operations.json failed AND CSV exists
        if (data is None or data.empty) and csv_file and os.path.exists(csv_file):
            try:
                data = pd.read_csv(csv_file)
                data_source = f"CSV file {csv_file}"
                if not title:
                    title = f"{plot_type.capitalize()} Plot from {os.path.basename(csv_file)}"
                click.echo(f"📊 Fallback: Loaded CSV data with {len(data)} rows and {len(data.columns)} columns")
                click.echo(f"📋 CSV columns: {list(data.columns)}")
            except Exception as e:
                click.echo(f"❌ Error reading CSV file {csv_file}: {e}")
                data = None
        
        # Error if no data found
        if data is None or data.empty:
            if not os.path.exists(data_file) and not os.path.exists(csv_file):
                click.echo(f"❌ Error: Neither '{data_file}' nor '{csv_file}' found")
            else:
                click.echo("❌ Error: No valid data found for plotting")
            return
        
        # Set appropriate labels based on operation type
        if operation_type:
            if not x_label:
                x_label = "Features" if operation_type in ['mean', 'std', 'range', 'median', 'mode'] else "Categories"
            if not y_label:
                y_label = get_y_label_for_operation(operation_type)
        
        click.echo(f"🎨 Creating {plot_type} plot from {data_source}")
        click.echo(f"📏 Plot dimensions: {width}x{height}, Theme: {theme}")
        
        # Create the plot using the create_plot function from plotter.py
        fig = create_plot(
            data=data,
            plot_type=plot_type,
            title=title,
            x_label=x_label,
            y_label=y_label,
            color_column=color_column,
            size_column=size_column,
            facet_column=facet_column,
            output_file=output,
            width=width,
            height=height,
            theme=theme
        )
        
        click.echo(f"✅ Successfully created {plot_type} plot!")
        click.echo("📊 Plot data saved to plotly.json")
        
        if show:
            click.echo("🌐 Opening plot in browser...")
            fig.show()
            
    except Exception as e:
        click.echo(f"❌ Error creating plot: {e}")
        import traceback
        click.echo("Full error details:")
        click.echo(traceback.format_exc())




@click.group()
def cli():
    pass

cli.add_command(download_data)
cli.add_command(observation)
cli.add_command(condition)
cli.add_command(correlation)
cli.add_command(abbreviate)
cli.add_command(mean)
cli.add_command(median)
cli.add_command(mode)
cli.add_command(std)
cli.add_command(range)
cli.add_command(frequency)
cli.add_command(cluster)
cli.add_command(plot)
cli.add_command(join)

if __name__ == '__main__':
    cli()