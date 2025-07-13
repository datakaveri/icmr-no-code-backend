SUPPORTED_PLOT_TYPES = [
    'bar', 'line', 'scatter', 'histogram', 'box', 'violin', 'heatmap', 'pie', 'network'
]

DEFAULT_PLOT_CONFIG = {
    'width': 800,
    'height': 600,
    'theme': 'plotly',
    'font_size': 12,
    'title_font_size': 16,
    'axis_font_size': 14,
    'font_family': 'Arial Black'
}

OPERATION_LABELS = {
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
    'symptom_pattern': 'Confidence/Support',
}

CLUSTERING_EXCLUDE_COLUMNS = ['Cluster', 'cluster_id', 'point_id', 'Size', 'Distinctness']

PLOT_LIMITS = {
    'bar_chart_max_items': 15,
    'pie_chart_max_items': 10,
    'cluster_max_points': 50,
    'top_n_default': 15
}

NETWORK_CONFIG = {
    'seed': 42,
    'edge_width': 2,
    'edge_color': '#888',
    'node_size': 20,
    'center_node_size': 15,
    'center_node_color': 'black',
    'center_node_symbol': 'diamond',
    'colorscale': 'Viridis'
}

PCA_CONFIG = {
    'n_components': 2,
    'standardize': True,
    'jitter_std': 0.1,
    'variation_factor': 0.1
}