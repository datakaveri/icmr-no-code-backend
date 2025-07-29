import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_box_plot(df, x_label, y_label, color_column, size_column, facet_column):
    if 'Cluster' in df.columns and any(col for col in df.columns if col not in ['Cluster', 'cluster_id', 'point_id']):
        numeric_cols = [col for col in df.columns if col not in ['Cluster', 'cluster_id', 'point_id'] and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) > 0:
            y_col = numeric_cols[0]
            fig = px.box(df, x='Cluster', y=y_col, 
                        title=f"Distribution of {y_col} by Cluster",
                        color='Cluster')
            
            fig.update_layout(
                xaxis_title="Clusters",
                yaxis_title=y_col,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
    
    if 'Feature' in df.columns:
        numeric_cols = [col for col in df.columns if col != 'Feature' and pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) > 0:
            y_col = numeric_cols[0]  # Use first numeric column
            
            # Create individual box plots for each feature
            fig = go.Figure()
            
            for i, feature in enumerate(df['Feature'].unique()):
                feature_data = df[df['Feature'] == feature][y_col]
                
                fig.add_trace(go.Box(
                    y=feature_data,
                    name=feature,
                    boxmean=True,  # Show mean line
                    boxpoints='outliers'  # Show outliers
                ))
            
            fig.update_layout(
                title=f"Distribution of {y_col} by Feature",
                xaxis_title="Features",
                yaxis_title=y_col,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
    
    # Handle frequency data
    if 'Category' in df.columns and ('Count' in df.columns or 'Frequency' in df.columns):
        y_col = 'Count' if 'Count' in df.columns else 'Frequency'
        
        fig = go.Figure()
        
        # Group data and create box plots
        categories = df['Category'].unique()[:10]  
        
        for category in categories:
            cat_data = df[df['Category'] == category][y_col]
            
            fig.add_trace(go.Box(
                y=cat_data,
                name=str(category)[:20],
                boxmean=True,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title=f"Distribution of {y_col} by Category",
            xaxis_title="Categories",
            yaxis_title=y_col,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    # Handle correlation data
    if 'Column_Pair' in df.columns:
        coef_cols = [col for col in df.columns if 'Coefficient' in col and pd.api.types.is_numeric_dtype(df[col])]
        if coef_cols:
            y_col = coef_cols[0]
            
            fig = px.box(df, y=y_col, 
                        title=f"Distribution of {y_col}",
                        boxpoints='outliers')
            
            fig.update_layout(
                yaxis_title=y_col,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig
    
    # Handle time series data
    if any('date' in col.lower() for col in df.columns):
        date_col = next((col for col in df.columns if 'date' in col.lower()), df.columns[0])
        numeric_cols = [col for col in df.columns if col != date_col and pd.api.types.is_numeric_dtype(df[col])]
        
        if numeric_cols:
            y_col = numeric_cols[0]
            
            # Convert date column to datetime
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Extract month-year for grouping
            df['period'] = df[date_col].dt.to_period('M').astype(str)
            
            fig = px.box(df, x='period', y=y_col, 
                        title=f"Monthly Distribution of {y_col}")
            
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title=y_col,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(tickangle=45)
            return fig
    
    # Default handling for any other data structure
    # Determine x and y columns
    if x_label and x_label in df.columns:
        x_col = x_label
    else:
        # Use first categorical or object column
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        x_col = cat_cols[0] if len(cat_cols) > 0 else df.columns[0]
    
    if y_label and y_label in df.columns:
        y_col = y_label
    else:
        # Use first numeric column that's not x_col
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != x_col]
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    # Create the box plot
    if y_col:
        fig = px.box(df, x=x_col, y=y_col, 
                    color=color_column if color_column and color_column in df.columns else None,
                    facet_col=facet_column if facet_column and facet_column in df.columns else None,
                    title=f"Distribution of {y_col} by {x_col}",
                    boxpoints='outliers')
    else:
        # If no separate y column, create box plot of x column values
        fig = px.box(df, y=x_col, 
                    color=color_column if color_column and color_column in df.columns else None,
                    title=f"Distribution of {x_col}",
                    boxpoints='outliers')
    
    # Update layout for better appearance
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linecolor='black'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            showline=True,
            linecolor='black'
        )
    )
    
    return fig