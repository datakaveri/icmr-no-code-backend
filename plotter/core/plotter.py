import plotly.io as pio
import pandas as pd
import json
import os
from datetime import datetime
from ..plots.bar import create_bar_plot
from ..plots.line import create_line_plot
from ..plots.scatter import create_scatter_plot
from ..plots.histogram import create_histogram_plot
from ..plots.box import create_box_plot
from ..plots.violin import create_violin_plot
from ..plots.heatmap import create_heatmap_plot
from ..plots.pie import create_pie_plot
from ..plots.network import create_network_plot
from .data_processor import prepare_data

class GenericPlotter:
    def __init__(self):
        self.supported_plots = {
            'bar': create_bar_plot,
            'line': create_line_plot,
            'scatter': create_scatter_plot,
            'histogram': create_histogram_plot,
            'box': create_box_plot,
            'violin': create_violin_plot,
            'heatmap': create_heatmap_plot,
            'pie': create_pie_plot,
            'network': create_network_plot 
        }
    
    def create_plot(self, data, plot_type, title=None, x_label=None, y_label=None, 
                   color_column=None, size_column=None, facet_column=None, 
                   output_file=None, width=800, height=600, theme='plotly'):
        
        if plot_type not in self.supported_plots:
            raise ValueError(f"Unsupported plot type: {plot_type}. Supported types: {list(self.supported_plots.keys())}")
        
        df = prepare_data(data)
        
        if df.empty:
            raise ValueError("No data available for plotting")
        
        fig = self.supported_plots[plot_type](df, x_label, y_label, color_column, size_column, facet_column)
        
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