import pandas as pd
import numpy as np
from ..analytics.statistical import StatisticalAnalytics
from ..analytics.clustering import ClusteringAnalytics
from ..analytics.correlation import CorrelationAnalytics
from ..analytics.pattern_analysis import PatternAnalytics
from ..analytics.report import ReportGenerator

class StatisticalOperations:
    def __init__(self, df):
        self.df = df
        self.statistical = StatisticalAnalytics(df)
        self.clustering = ClusteringAnalytics(df)
        self.correlation = CorrelationAnalytics(df)
        self.pattern = PatternAnalytics(df)
        self.report_generator = ReportGenerator(df)
    
    def calculate_mean(self, column=None):
        return self.statistical.calculate_mean(column)
    
    def calculate_median(self, column=None):
        return self.statistical.calculate_median(column)
    
    def calculate_mode(self, column=None):
        return self.statistical.calculate_mode(column)
    
    def calculate_std(self, column=None):
        return self.statistical.calculate_std(column)
    
    def calculate_range(self, column=None):
        return self.statistical.calculate_range(column)
    
    def frequency_analysis(self, column=None, proportion=False):
        return self.statistical.frequency_analysis(column, proportion)
    
    def patient_segmentation(self, groupby_col, obs_names_path='obs_names.pkl', cond_names_path='cond_names.pkl', top_n=5):
        return self.statistical.patient_segmentation(groupby_col, obs_names_path, cond_names_path, top_n)
    
    def perform_clustering(self, features=None, clusters=3, topx=3, segment_clusters=False, obs_names_path=None, cond_names_path=None, top_n=5):
        return self.clustering.perform_clustering(features, clusters, topx, segment_clusters, obs_names_path, cond_names_path, top_n)
    
    def calculate_prevalence(self, df, disease_col, case_value=1):
        return self.statistical.calculate_prevalence(df, disease_col, case_value)
    
    def correlation_coefficients(self, df, col1=None, col2=None):
        return self.correlation.correlation_coefficients(df, col1, col2)
    
    def symptom_pattern_analysis_analysis(self, obs_names_path='obs_names.pkl', cond_names_path='cond_names.pkl', min_support=0.1, min_confidence=0.7, min_lift=1.2, exclude_cols=None, include_network_analysis=True, export_csv=False):
        return self.pattern.symptom_pattern_analysis_analysis(obs_names_path, cond_names_path, min_support, min_confidence, min_lift, exclude_cols, include_network_analysis, export_csv)
    
    def calculate_covariance(self, df, col1=None, col2=None):
        return self.correlation.calculate_covariance(df, col1, col2)
    
    def generate_report(self, operations_file='operations.json', plots_file='plotly.json', output_dir='reports', render_html=True, open_browser=False):
        return self.report_generator.generate_report(operations_file, plots_file, output_dir, render_html, open_browser)