import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import sys
from pathlib import Path

class ReportGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_name = f"healthcare_analytics_report_{self.report_timestamp}"
        
    def _json_serialize(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._json_serialize(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._json_serialize(item) for item in obj]
        else:
            return obj
    
    def _load_operations_data(self, operations_file: str) -> Dict[str, Any]:
        """Load and process operations data from JSON file"""
        if not os.path.exists(operations_file):
            return {"results": [], "summary": {"total_operations": 0}}
        
        try:
            with open(operations_file, 'r') as f:
                data = json.load(f)
            return self._json_serialize(data)
        except Exception as e:
            print(f"Warning: Could not load operations data: {e}")
            return {"results": [], "summary": {"total_operations": 0}}
    
    def _load_plots_data(self, plots_file: str) -> Dict[str, Any]:
        """Load plotly plots data from JSON file"""
        if not os.path.exists(plots_file):
            return {"plots": []}
        
        try:
            with open(plots_file, 'r') as f:
                data = json.load(f)
            return self._json_serialize(data)
        except Exception as e:
            print(f"Warning: Could not load plots data: {e}")
            return {"plots": []}
    
    def _generate_dataset_overview(self) -> Dict[str, Any]:
        """Generate basic dataset overview"""
        overview = {
            "basic_info": {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                "columns": list(self.df.columns)
            },
            "data_types": {
                "numeric": list(self.df.select_dtypes(include=[np.number]).columns),
                "categorical": list(self.df.select_dtypes(include=['object', 'category']).columns),
                "datetime": list(self.df.select_dtypes(include=['datetime64']).columns)
            }
        }
        
        return self._json_serialize(overview)
    
    def _create_quarto_document(self, operations_data: Dict, plots_data: Dict, 
                              dataset_overview: Dict, output_dir: str) -> str:
        """Create Quarto document with all analytics results"""
        
        qmd_content = f"""---
title: "Healthcare Analytics Report"
author: "Analytics SDK"
date: "{datetime.now().strftime('%B %d, %Y')}"
format:
  html:
    theme: flatly
    toc: true
    toc-location: left
    toc-depth: 2
    code-fold: true
    code-summary: "Show code"
    embed-resources: true
    page-layout: full
    fig-width: 12
    fig-height: 8
    css: styles.css
    grid:
      sidebar-width: 280px
      body-width: 1100px
      margin-width: 180px
execute:
  echo: false
  warning: false
  message: false
---

```{{python}}
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import plotly.io as pio
from IPython.display import HTML, display

# Load data
with open('report_data.json', 'r') as f:
    report_data = json.load(f)

operations_data = report_data['operations']
plots_data = report_data['plots']
dataset_overview = report_data['dataset_overview']
```

```{{python}}
operations_results = operations_data.get('results', [])
plots_list = plots_data.get('plots', [])

# Create a mapping of operations to their corresponding plots
operation_plots = {{}}
for i, plot in enumerate(plots_list):
    plot_name = plot.get('name', f'Plot {{i+1}}')
    plot_type = plot.get('type', 'unknown')
    
    # Try to match plots with operations based on naming or content
    for op in operations_results:
        op_name = op['operation']
        if op_name.lower() in plot_name.lower() or plot_name.lower() in op_name.lower():
            if op_name not in operation_plots:
                operation_plots[op_name] = []
            operation_plots[op_name].append(plot)

print("Processing operations and their visualizations...")
```

## Analysis Results

```{{python}}
for i, operation in enumerate(operations_results, 1):
    operation_name = operation['operation']
    operation_data = operation.get('data', {{}})
    operation_timestamp = operation.get('timestamp', 'N/A')
    
    print(f"\\n")
    print(f"{{i}}. {{operation_name.upper().replace('_', ' ')}} ANALYSIS")
    print(f"Executed at: {{operation_timestamp}}")
    print(f"\\n")
    
    # Display operation-specific results
    if operation_name == 'download_data':
        print("Data Download Summary")
        if isinstance(operation_data, dict):
            for key, value in operation_data.items():
                if isinstance(value, (list, dict)):
                    print(f"{{key.title()}}: {{len(value)}} items")
                else:
                    print(f"{{key.title()}}: {{value}}")
        else:
            print("Data download completed successfully")
    
    elif operation_name in ['mean', 'median', 'mode', 'std', 'range']:
        print(f"{{operation_name.title()}} Statistics")
        if isinstance(operation_data, dict) and operation_data:
            # Create DataFrame for better display
            stats_df = pd.DataFrame(list(operation_data.items()), columns=['Feature', operation_name.title()])
            # Sort by value (descending for most metrics)
            if operation_name != 'mode':
                stats_df = stats_df.sort_values(operation_name.title(), ascending=False)
            
            print(f"\\nTop 10 Results:")
            print(stats_df.head(10).to_markdown(index=False))
        else:
            print(f"Result: {{operation_data}}")
    
    elif operation_name == 'frequency':
        print("Frequency Analysis")
        if isinstance(operation_data, dict) and operation_data:
            freq_df = pd.DataFrame(list(operation_data.items()), columns=['Category', 'Frequency'])
            freq_df = freq_df.sort_values('Frequency', ascending=False)
            
            print(f"\\nTop 15 Most Frequent Categories:")
            print(freq_df.head(15).to_markdown(index=False))
    
    elif operation_name == 'correlation':
        print("Correlation Analysis")
        if isinstance(operation_data, dict) and operation_data:
            corr_df = pd.DataFrame(list(operation_data.items()), columns=['Feature Pair', 'Correlation'])
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            print(f"\\nTop 15 Correlations:")
            print(corr_df.head(15).to_markdown(index=False))
    
    elif operation_name == 'cluster':
        print("Clustering Analysis")
        if isinstance(operation_data, dict):
            if 'cluster_summary' in operation_data:
                cluster_summary = operation_data['cluster_summary']
                print(f"\\nNumber of Clusters Identified: {{len(cluster_summary)}}")
                
                cluster_df = pd.DataFrame(cluster_summary).T
                print(f"\\nCluster Distribution:")
                print(cluster_df.to_markdown())
            else:
                print("Clustering analysis completed successfully")
                for key, value in operation_data.items():
                    if isinstance(value, (list, dict)):
                        print(f"{{key.title()}}: {{len(value)}} items")
                    else:
                        print(f"{{key.title()}}: {{value}}")
    
    elif operation_name == 'prevalence':
        print("Prevalence Analysis")
        if isinstance(operation_data, dict) and operation_data:
            prev_df = pd.DataFrame(list(operation_data.items()), columns=['Condition', 'Prevalence'])
            prev_df = prev_df.sort_values('Prevalence', ascending=False)
            
            print(f"\\nCondition Prevalence Rates:")
            print(prev_df.to_markdown(index=False))
    
    elif operation_name == 'symptom_pattern':
        print("Symptom Pattern Analysis")
        if isinstance(operation_data, dict):
            if 'association_rules' in operation_data:
                rules = operation_data['association_rules'].get('rules', [])
                if rules:
                    print(f"\\nAssociation Rules Discovered: {{len(rules)}}")
                    
                    print(f"\\nTop 10 Association Rules:")
                    for j, rule in enumerate(rules[:10], 1):
                        print(f"{{j}}. {{rule.get('antecedents', '')}} → {{rule.get('consequents', '')}}")
                        print(f"   Support: {{rule.get('support', 0):.3f}}, Confidence: {{rule.get('confidence', 0):.3f}}, Lift: {{rule.get('lift', 0):.3f}}")
            
            if 'network_analysis' in operation_data:
                network = operation_data['network_analysis']
                print(f"\\nNetwork Analysis Results:")
                print(f"Total Nodes: {{network.get('total_nodes', 0)}}")
                print(f"Total Edges: {{network.get('total_edges', 0)}}")
                
                if 'most_central_nodes' in network:
                    print(f"\\nMost Central Nodes:")
                    for node in network['most_central_nodes'][:5]:
                        print(f"  {{node.get('node', 'N/A')}} ({{node.get('type', 'N/A')}}): {{node.get('degree_centrality', 0):.3f}}")
    
    else:
        # Generic handling for other operations
        print(f"{{operation_name.title().replace('_', ' ')}} Results")
        if isinstance(operation_data, dict) and operation_data:
            for key, value in operation_data.items():
                if isinstance(value, (list, dict)):
                    print(f"{{key.title()}}: {{len(value)}} items")
                else:
                    print(f"{{key.title()}}: {{value}}")
        elif isinstance(operation_data, list) and operation_data:
            if len(operation_data) > 0 and isinstance(operation_data[0], dict):
                df_display = pd.DataFrame(operation_data)
                print(df_display.head(10).to_markdown(index=False))
            else:
                print("\\n".join([f"{{item}}" for item in operation_data[:10]]))
        else:
            print(f"Operation completed successfully")
    
    # Display related plots
    if operation_name in operation_plots:
        print(f"\\nVisualizations")
        for plot_info in operation_plots[operation_name]:
            plot_name = plot_info.get('name', 'Visualization')
            plot_type = plot_info.get('type', 'unknown')
            
            print(f"\\n{{plot_name}} ({{plot_type}})")
            
            # Display the actual Plotly plot
            if 'data' in plot_info:
                try:
                    plot_data = plot_info['data']
                    if isinstance(plot_data, str):
                        # If data is a JSON string, parse it
                        plot_data = json.loads(plot_data)
                    
                    # Create Plotly figure from the data
                    if isinstance(plot_data, dict):
                        # Check if it's a complete figure dict
                        if 'data' in plot_data and 'layout' in plot_data:
                            fig = go.Figure(data=plot_data['data'], layout=plot_data['layout'])
                        else:
                            fig = go.Figure(plot_data)
                        
                        # Update layout for better display
                        fig.update_layout(
                            height=500,
                            showlegend=True,
                            title_font_size=16,
                            template='plotly_white'
                        )
                        
                        fig.show()
                    else:
                        print("Plot data format not recognized")
                        
                except Exception as e:
                    print(f"Could not display plot: {{str(e)}}")
            else:
                print("No plot data available")
    
    print(f"\\n")
    print("---")
```

## Summary & Insights

```{{python}}
# Generate summary statistics
print("Analysis Summary")
print(f"\\nDataset Information:")
print(f"Total records analyzed: {{dataset_overview['basic_info']['total_rows']:,d}}")
print(f"Features examined: {{dataset_overview['basic_info']['total_columns']:,d}}")
print(f"Memory footprint: {{dataset_overview['basic_info']['memory_usage']}}")

print(f"\\nOperations Executed:")
operation_counts = {{}}
for op in operations_results:
    op_name = op['operation']
    operation_counts[op_name] = operation_counts.get(op_name, 0) + 1

for op_name, count in sorted(operation_counts.items()):
    print(f"{{op_name.replace('_', ' ').title()}}: {{count}} time(s)")

print(f"\\nVisualizations Generated:")
print(f"Total plots created: {{len(plots_list)}}")

plot_types = {{}}
for plot in plots_list:
    plot_type = plot.get('type', 'unknown')
    plot_types[plot_type] = plot_types.get(plot_type, 0) + 1

for plot_type, count in sorted(plot_types.items()):
    print(f"{{plot_type.title()}}: {{count}} plot(s)")
```

---

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
Analysis Framework: v1.0.0
"""
        
        # Save the Quarto document
        qmd_file = os.path.join(output_dir, f"{self.report_name}.qmd")
        with open(qmd_file, 'w', encoding='utf-8') as f:
            f.write(qmd_content)
        
        return qmd_file
    
    def _create_custom_css(self, output_dir: str) -> str:
        """Create custom CSS matching the exact styling from the image"""
        css_content = """
/* Healthcare Analytics Report - Exact Image Styling */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --primary-blue: #6699CC;
  --secondary-blue: #4A90C2;
  --accent-blue: #7AA8D1;
  --text-primary: #2C3E50;
  --text-secondary: #34495E;
  --text-muted: #7F8C8D;
  --bg-primary: #ffffff;
  --bg-secondary: #F8F9FA;
  --bg-light: #FAFBFC;
  --border-color: #E9ECEF;
  --border-light: #F1F3F4;
}

/* Reset and Base Styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
  font-size: 15px;
  line-height: 1.6;
  color: var(--text-secondary);
  background-color: var(--bg-primary);
  font-weight: 400;
}

/* Typography - Exact Match to Image */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  color: var(--primary-blue);
  margin-bottom: 1.5rem;
  line-height: 1.3;
}

h1 {
  font-size: 2.25rem;
  font-weight: 700;
  color: var(--primary-blue);
  margin-bottom: 0.75rem;
}

h2 {
  font-size: 1.5rem;
  color: var(--primary-blue);
  margin-top: 2rem;
  margin-bottom: 1.5rem;
  font-weight: 600;
}

h3 {
  font-size: 1.25rem;
  color: var(--text-primary);
  margin-top: 1.5rem;
  margin-bottom: 1rem;
  font-weight: 500;
}

h4 {
  font-size: 1.125rem;
  color: var(--text-primary);
  margin-top: 1.25rem;
  margin-bottom: 0.75rem;
  font-weight: 500;
}

p {
  margin-bottom: 1rem;
  color: var(--text-secondary);
  font-size: 15px;
  line-height: 1.65;
  font-weight: 400;
}

/* Strong text styling to match image */
strong {
  color: var(--text-primary);
  font-weight: 600;
}

/* Code blocks - cleaner styling */
pre {
  background-color: var(--bg-light);
  border: 1px solid var(--border-light);
  border-radius: 6px;
  padding: 1.25rem;
  overflow-x: auto;
  font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
  font-size: 13px;
  line-height: 1.5;
  margin: 1.5rem 0;
  color: var(--text-primary);
}

code {
  background-color: var(--bg-light);
  color: var(--text-primary);
  padding: 0.25rem 0.5rem;
  border-radius: 3px;
  font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
  font-size: 13px;
  border: 1px solid var(--border-light);
}

/* Table styling to match image exactly */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.5rem 0;
  background: var(--bg-primary);
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--border-color);
  font-size: 14px;
}

th {
  background-color: var(--bg-secondary);
  color: var(--text-primary);
  padding: 12px 16px;
  text-align: left;
  font-weight: 600;
  font-size: 14px;
  border-bottom: 1px solid var(--border-color);
}

td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-light);
  color: var(--text-secondary);
  vertical-align: top;
  font-size: 14px;
}

tbody tr:hover {
  background-color: var(--bg-light);
  transition: background-color 0.2s ease;
}

/* Layout styling */
.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 2rem;
}

/* Sidebar styling to match image */
.sidebar {
  background-color: var(--bg-secondary);
  padding: 1.5rem;
  font-size: 14px;
}

.sidebar h3 {
  color: var(--text-primary);
  font-size: 16px;
  margin-bottom: 1rem;
  font-weight: 600;
}

.sidebar ul {
  list-style: none;
  padding: 0;
}

.sidebar li {
  margin-bottom: 0.5rem;
}

.sidebar a {
  color: var(--text-secondary);
  text-decoration: none;
  padding: 0.5rem 0;
  display: block;
  font-size: 14px;
  transition: color 0.2s ease;
}

.sidebar a:hover {
  color: var(--primary-blue);
}

/* Lists */
ul, ol {
  padding-left: 1.5rem;
  margin: 1rem 0;
}

li {
  margin-bottom: 0.5rem;
  color: var(--text-secondary);
  line-height: 1.6;
  font-size: 15px;
}

/* Plotly charts */
.plotly-graph-div {
  border: 1px solid var(--border-color);
  border-radius: 6px;
  margin: 1.5rem 0;
  overflow: hidden;
  background: var(--bg-primary);
}

/* Horizontal rules */
hr {
  border: none;
  height: 1px;
  background: var(--border-color);
  margin: 2rem 0;
}

/* Text formatting */
.timestamp {
  color: var(--text-muted);
  font-size: 13px;
  font-style: italic;
}

/* Responsive design */
@media (max-width: 768px) {
  body {
    font-size: 14px;
  }
  
  .container {
    padding: 0 1rem;
  }
  
  table {
    font-size: 12px;
  }
  
  th, td {
    padding: 10px 12px;
  }
  
  h1 {
    font-size: 1.875rem;
  }
  
  h2 {
    font-size: 1.25rem;
  }
}

/* Print styles */
@media print {
  body {
    background: white;
    color: black;
    font-size: 12pt;
  }
  
  .plotly-graph-div {
    page-break-inside: avoid;
  }
}

/* Selection */
::selection {
  background: var(--primary-blue);
  color: white;
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
  background: var(--primary-blue);
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--secondary-blue);
}
"""
        
        css_file = os.path.join(output_dir, "styles.css")
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        return css_file
    
    def generate_report(self, operations_file: str = "operations.json", 
                       plots_file: str = "plotly.json", 
                       output_dir: str = "reports",
                       render_html: bool = True,
                       open_browser: bool = False) -> Dict[str, Any]:
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Load all data
            operations_data = self._load_operations_data(operations_file)
            plots_data = self._load_plots_data(plots_file)
            dataset_overview = self._generate_dataset_overview()
            
            # Combine all data
            report_data = {
                "operations": operations_data,
                "plots": plots_data,
                "dataset_overview": dataset_overview,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "report_name": self.report_name,
                    "sdk_version": "1.0.0"
                }
            }
            
            # Save combined data for Quarto
            report_data_file = os.path.join(output_dir, "report_data.json")
            with open(report_data_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            # Create custom CSS
            css_file = self._create_custom_css(output_dir)
            
            # Create Quarto document
            qmd_file = self._create_quarto_document(operations_data, plots_data, dataset_overview, output_dir)
            
            result = {
                "success": True,
                "report_name": self.report_name,
                "files_created": {
                    "quarto_document": qmd_file,
                    "data_file": report_data_file,
                    "style_file": css_file
                },
                "summary": {
                    "total_operations": len(operations_data.get('results', [])),
                    "total_plots": len(plots_data.get('plots', [])),
                    "dataset_rows": dataset_overview['basic_info']['total_rows'],
                    "dataset_columns": dataset_overview['basic_info']['total_columns']
                }
            }
            
            # Render HTML if requested
            if render_html:
                try:
                    # Change to output directory for rendering
                    original_dir = os.getcwd()
                    os.chdir(output_dir)
                    
                    # Render Quarto document
                    render_cmd = ["quarto", "render", f"{self.report_name}.qmd"]
                    render_result = subprocess.run(render_cmd, capture_output=True, text=True)
                    
                    if render_result.returncode == 0:
                        html_file = os.path.join(output_dir, f"{self.report_name}.html")
                        result["files_created"]["html_report"] = html_file
                        result["render_success"] = True
                        
                        # Open in browser if requested
                        if open_browser and os.path.exists(html_file):
                            import webbrowser
                            webbrowser.open(f"file://{os.path.abspath(html_file)}")
                            result["browser_opened"] = True
                    else:
                        result["render_success"] = False
                        result["render_error"] = render_result.stderr
                    
                    # Return to original directory
                    os.chdir(original_dir)
                    
                except Exception as e:
                    result["render_success"] = False
                    result["render_error"] = str(e)
                    if 'original_dir' in locals():
                        os.chdir(original_dir)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }