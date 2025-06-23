import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import json
import os
import click
from datetime import datetime

class GenericPlotter:
    def __init__(self):
        self.supported_plots = {
            'bar': self._create_bar_plot,
            'line': self._create_line_plot,
            'scatter': self._create_scatter_plot,
            'histogram': self._create_histogram_plot,
            'box': self._create_box_plot,
            'violin': self._create_violin_plot,
            'heatmap': self._create_heatmap_plot,
            'pie': self._create_pie_plot
        }
    
    def create_plot(self, data, plot_type, title=None, x_label=None, y_label=None, 
                   color_column=None, size_column=None, facet_column=None, 
                   output_file=None, width=800, height=600, theme='plotly'):
        
        if plot_type not in self.supported_plots:
            raise ValueError(f"Unsupported plot type: {plot_type}. Supported types: {list(self.supported_plots.keys())}")
        
        # Convert data to DataFrame if it's a dictionary
        df = self._prepare_data(data)
        
        if df.empty:
            raise ValueError("No data available for plotting")
        
        # Create the plot
        fig = self.supported_plots[plot_type](df, x_label, y_label, color_column, size_column, facet_column)
        
        # Update layout
        fig.update_layout(
            title=title or f"{plot_type.capitalize()} Plot",
            width=width,
            height=height,
            template=theme,
            showlegend=True,
            font=dict(size=12),
            title_font=dict(size=16, family="Arial Black"),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14)
        )
        
        # Always generate PNG file
        png_file = self._generate_png_filename(output_file, plot_type, title)
        try:
            pio.write_image(fig, png_file)
            print(f"PNG file saved: {png_file}")
        except Exception as e:
            print(f"Warning: Could not save PNG file: {e}")
        
        self._save_plot_to_json(fig, plot_type, title, png_file)
        
        return fig
    
    def _generate_png_filename(self, output_file, plot_type, title):
        if output_file:
            if not output_file.lower().endswith('.png'):
                output_file += '.png'
            return output_file
        else:
            safe_title = title.replace(' ', '_').replace('/', '_') if title else plot_type
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{safe_title}_{timestamp}.png"
    
    def _prepare_data(self, data):
        """Convert various data formats to DataFrame - handles ALL statistical operations"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            # Handle different statistical operation results
            
            # Standard Deviation results
            if 'stds' in data:
                items = list(data['stds'].items())
                # Sort by standard deviation value (descending)
                items.sort(key=lambda x: x[1], reverse=True)
                return pd.DataFrame(items, columns=['Feature', 'Standard_Deviation'])
            
            # Mean results
            elif 'means' in data:
                items = list(data['means'].items())
                # Sort by absolute mean value (descending)
                items.sort(key=lambda x: abs(x[1]), reverse=True)
                return pd.DataFrame(items, columns=['Feature', 'Mean'])
            
            # Median results
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
                    
                    # Handle multiple modes
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
                # Sort by range value (descending)
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
                # Sort by count (descending)
                freq_data.sort(key=lambda x: x['Count'], reverse=True)
                return pd.DataFrame(freq_data)
            
            elif 'top_clusters' in data:
                cluster_data = []
                for cluster_id, info in data['top_clusters'].items():
                    cluster_data.append({
                        'Cluster': f"Cluster {cluster_id}",
                        'Size': info['size'],
                        'Distinctness': info['distinctness']
                    })
                return pd.DataFrame(cluster_data)
            
            # Single operation results
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
            
            # Generic key-value pairs - this handles simple dictionaries like condition/observation data
            else:
                items = list(data.items())
                # Try to sort numerically if possible
                try:
                    items.sort(key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
                except:
                    pass
                return pd.DataFrame(items, columns=['Category', 'Value'])
        else:
            raise ValueError("Data must be a pandas DataFrame or dictionary")
    
    def _create_bar_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        # Get appropriate columns
        x_col = x_label if x_label and x_label in df.columns else df.columns[0]
        y_col = y_label if y_label and y_label in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Limit to top 15 items for readability
        if len(df) > 15:
            df = df.head(15)
        
        fig = px.bar(df, x=x_col, y=y_col, color=color_column, 
                    facet_col=facet_column, hover_data=df.columns.tolist())
        
        # Rotate x-axis labels if they're too long
        if df[x_col].dtype == 'object':
            max_label_length = max(len(str(label)) for label in df[x_col])
            if max_label_length > 10:
                fig.update_xaxes(tickangle=45)
        
        return fig
    
    def _create_line_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        x_col = x_label if x_label and x_label in df.columns else df.columns[0]
        y_col = y_label if y_label and y_label in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        fig = px.line(df, x=x_col, y=y_col, color=color_column, 
                     facet_col=facet_column, hover_data=df.columns.tolist())
        
        return fig
    
    def _create_scatter_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        x_col = x_label if x_label and x_label in df.columns else df.columns[0]
        y_col = y_label if y_label and y_label in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_column, size=size_column,
                        facet_col=facet_column, hover_data=df.columns.tolist())
        
        return fig
    
    def _create_histogram_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        x_col = x_label if x_label and x_label in df.columns else df.columns[0]
        
        fig = px.histogram(df, x=x_col, color=color_column, facet_col=facet_column,
                          hover_data=df.columns.tolist())
        
        return fig
    
    def _create_box_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        x_col = x_label if x_label and x_label in df.columns else df.columns[0]
        y_col = y_label if y_label and y_label in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        
        if y_col:
            fig = px.box(df, x=x_col, y=y_col, color=color_column, facet_col=facet_column,
                        hover_data=df.columns.tolist())
        else:
            fig = px.box(df, y=x_col, color=color_column, facet_col=facet_column,
                        hover_data=df.columns.tolist())
        
        return fig
    
    def _create_violin_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        x_col = x_label if x_label and x_label in df.columns else df.columns[0]
        y_col = y_label if y_label and y_label in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        
        if y_col:
            fig = px.violin(df, x=x_col, y=y_col, color=color_column, facet_col=facet_column,
                           hover_data=df.columns.tolist())
        else:
            fig = px.violin(df, y=x_col, color=color_column, facet_col=facet_column,
                           hover_data=df.columns.tolist())
        
        return fig
    
    def _create_heatmap_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        if len(df.columns) >= 3:
            pivot_df = df.pivot_table(index=df.columns[0], columns=df.columns[1], 
                                     values=df.columns[2], aggfunc='mean')
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                pivot_df = df.set_index(df.columns[0])[numeric_cols]
            else:
                pivot_df = df.corr()
        
        fig = px.imshow(pivot_df, aspect="auto", color_continuous_scale='Viridis')
        
        return fig
    
    def _create_pie_plot(self, df, x_label, y_label, color_column, size_column, facet_column):
        names_col = df.columns[0]
        values_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Limit to top 10 items for pie chart readability
        if len(df) > 10:
            df = df.head(10)
        
        if len(df.columns) == 1 or df[values_col].dtype == 'object':
            # Count occurrences if only one column or values are non-numeric
            fig = px.pie(df, names=names_col, 
                        hover_data=df.columns.tolist())
        else:
            fig = px.pie(df, names=names_col, values=values_col, 
                        hover_data=df.columns.tolist())
        
        return fig
    
    def _save_plot_to_json(self, fig, plot_type, plot_name=None, png_file=None):
        plotly_file = 'plotly.json'
        
        if os.path.exists(plotly_file):
            with open(plotly_file, 'r') as f:
                plot_data = json.load(f)
        else:
            plot_data = {'plots': []}
        
        plot_entry = {
            'type': plot_type,
            'data': pio.to_json(fig),
            'created_at': datetime.now().isoformat(),
            'png_file': png_file
        }
        
        if plot_name:
            plot_entry['name'] = plot_name
        
        plot_data['plots'].append(plot_entry)
        
        with open(plotly_file, 'w') as f:
            json.dump(plot_data, f, indent=2)

# Standalone functions for external use
def convert_operation_data_to_df(data, operation_type, top_n=15):
    """Convert operation results to a DataFrame suitable for plotting"""
    
    if operation_type == 'mean':
        if 'means' in data:
            means_data = data['means']
            # Sort by absolute value and take top N
            sorted_means = sorted(means_data.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            df = pd.DataFrame(sorted_means, columns=['Feature', 'Mean'])
            return df
    
    elif operation_type == 'std':
        if 'stds' in data:
            stds_data = data['stds']
            # Sort by value and take top N
            sorted_stds = sorted(stds_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_stds, columns=['Feature', 'Standard_Deviation'])
            return df
    
    elif operation_type == 'median':
        if 'medians' in data:
            medians_data = data['medians']
            # Sort by absolute value and take top N
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
                
                # Handle multiple modes
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
            # Extract range values and sort
            range_items = [(k, v['range'], v['min'], v['max']) for k, v in ranges_data.items()]
            sorted_ranges = sorted(range_items, key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_ranges, columns=['Feature', 'Range', 'Min', 'Max'])
            return df
    
    elif operation_type == 'frequency':
        if 'frequencies' in data:
            # Handle multiple columns of frequency data
            all_freq_data = []
            for col, freq_dict in data['frequencies'].items():
                for category, count in freq_dict.items():
                    all_freq_data.append({'Column': col, 'Category': category, 'Frequency': count})
            df = pd.DataFrame(all_freq_data[:top_n])
            return df
        elif 'frequency' in data:
            # Handle single column frequency data
            freq_data = data['frequency']
            sorted_freq = sorted(freq_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_freq, columns=['Category', 'Frequency'])
            return df
    
    elif operation_type == 'clustering':
        if 'top_clusters' in data:
            cluster_data = []
            for cluster_id, info in data['top_clusters'].items():
                cluster_data.append({
                    'Cluster': f"Cluster {cluster_id}",
                    'Size': info['size'],
                    'Distinctness': info['distinctness']
                })
            df = pd.DataFrame(cluster_data)
            return df
    
    elif operation_type == 'observation':
        # Handle observation data - it's a simple dictionary of observation: count
        if isinstance(data, dict):
            sorted_obs = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_obs, columns=['Observation', 'Count'])
            return df
    
    elif operation_type == 'condition':
        # IMPROVED: Handle condition data properly
        if isinstance(data, dict):
            # The data is already in the format: condition_name: count
            sorted_cond = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            df = pd.DataFrame(sorted_cond, columns=['Condition', 'Count'])
            
            # Clean up condition names for better display
            df['Condition_Clean'] = df['Condition'].apply(lambda x: x.replace('cond_', '').replace(' | ', ' + '))
            
            return df
    
    elif operation_type == 'correlation':
        # Handle correlation data - it's a list of dictionaries with Observation, Condition, Correlation
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            # Filter out NaN correlations and sort by absolute correlation
            df = df.dropna(subset=['Correlation'])
            df = df.reindex(df['Correlation'].abs().sort_values(ascending=False).index)
            return df[:top_n]
    
    # ENHANCED: Generic handling for various data formats
    if isinstance(data, dict):
        # Check if it's a simple key-value dictionary (like condition/observation results)
        if all(isinstance(v, (int, float)) for v in data.values()):
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Try to determine appropriate column names based on operation type
            if operation_type in ['condition', 'observation']:
                col_names = [operation_type.capitalize(), 'Count']
            elif operation_type == 'frequency':
                col_names = ['Category', 'Frequency']
            else:
                col_names = ['Category', 'Value']
            
            df = pd.DataFrame(sorted_items, columns=col_names)
            return df
        
        # Try to find the main data key for complex nested data
        for key in ['means', 'medians', 'modes', 'stds', 'ranges']:
            if key in data:
                items = list(data[key].items())[:top_n]
                df = pd.DataFrame(items, columns=['Feature', key.capitalize().rstrip('s')])
                return df
    
    # If we get here, we couldn't parse the data
    print(f"Warning: Could not parse data for operation type '{operation_type}'")
    print(f"Data type: {type(data)}")
    print(f"Data sample: {str(data)[:200]}...")
    
    return pd.DataFrame()

def get_y_label_for_operation(operation_type):
    labels = {
        'mean': 'Mean Value',
        'median': 'Median Value',
        'mode': 'Mode Value',
        'std': 'Standard Deviation',
        'range': 'Range',
        'frequency': 'Frequency/Count',
        'clustering': 'Cluster Size',
        'observation': 'Count',
        'condition': 'Count',
        'correlation': 'Correlation Value'
    }
    return labels.get(operation_type, 'Value')

def create_plot(data, plot_type, **kwargs):
    plotter = GenericPlotter()
    return plotter.create_plot(data, plot_type, **kwargs)