import plotly.express as px

def create_line_plot(df, x_label, y_label, color_column, size_column, facet_column):
    # For prevalence analysis over time
    if 'prevalence' in df.columns or 'Prevalence' in df.columns:
        prev_col = 'prevalence' if 'prevalence' in df.columns else 'Prevalence'
        x_col = x_label if x_label and x_label in df.columns else df.columns[0]
        
        fig = px.line(df, x=x_col, y=prev_col, 
                     title="Prevalence Analysis Over Time",
                     markers=True)
        
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title="Prevalence (%)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        return fig
    
    # Regular line plot
    x_col = x_label if x_label and x_label in df.columns else df.columns[0]
    y_col = y_label if y_label and y_label in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    fig = px.line(df, x=x_col, y=y_col, color=color_column, 
                 facet_col=facet_column, hover_data=df.columns.tolist())
    
    return fig