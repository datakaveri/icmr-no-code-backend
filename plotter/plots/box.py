import plotly.express as px
import pandas as pd

def create_box_plot(df, x_label, y_label, color_column, size_column, facet_column):
    # For time series data (like your second image)
    if 'date' in df.columns or 'Date' in df.columns or any('date' in col.lower() for col in df.columns):
        date_col = next((col for col in df.columns if 'date' in col.lower()), df.columns[0])
        value_col = y_label if y_label and y_label in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract month-year for grouping
        df['month_year'] = df[date_col].dt.to_period('M').astype(str)
        
        fig = px.box(df, x='month_year', y=value_col, 
                    title=f"Monthly Distribution of {value_col}")
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title=value_col,
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
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    # Regular box plot
    x_col = x_label if x_label and x_label in df.columns else df.columns[0]
    y_col = y_label if y_label and y_label in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    
    if y_col:
        fig = px.box(df, x=x_col, y=y_col, color=color_column, facet_col=facet_column,
                    hover_data=df.columns.tolist())
    else:
        fig = px.box(df, y=x_col, color=color_column, facet_col=facet_column,
                    hover_data=df.columns.tolist())
    
    return fig