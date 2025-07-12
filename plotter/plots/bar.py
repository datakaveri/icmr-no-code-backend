import plotly.express as px

def create_bar_plot(df, x_label, y_label, color_column, size_column, facet_column):
    # For statistical results (mean, median, mode, std, range, frequency)
    if 'statistic' in df.columns or 'Statistic' in df.columns:
        stat_col = 'statistic' if 'statistic' in df.columns else 'Statistic'
        value_col = 'value' if 'value' in df.columns else df.columns[1]
        
        fig = px.bar(df, x=stat_col, y=value_col, 
                    title="Statistical Analysis Results",
                    color=stat_col,
                    color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(
            xaxis_title="Statistic",
            yaxis_title="Value",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black'),
            showlegend=False
        )
        
        return fig
    
    # For frequency analysis
    if 'frequency' in df.columns or 'count' in df.columns:
        category_col = df.columns[0]
        freq_col = 'frequency' if 'frequency' in df.columns else 'count'
        
        # Limit to top 15 for readability
        if len(df) > 15:
            df = df.nlargest(15, freq_col)
        
        fig = px.bar(df, x=category_col, y=freq_col,
                    title="Frequency Distribution",
                    color=freq_col,
                    color_continuous_scale='Blues')
        
        fig.update_layout(
            xaxis_title=category_col,
            yaxis_title="Frequency",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )
        
        # Rotate labels if they're long
        if df[category_col].dtype == 'object':
            max_label_length = max(len(str(label)) for label in df[category_col])
            if max_label_length > 10:
                fig.update_xaxes(tickangle=45)
        
        return fig
    
    # Regular bar plot
    x_col = x_label if x_label and x_label in df.columns else df.columns[0]
    y_col = y_label if y_label and y_label in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    if len(df) > 15:
        df = df.head(15)
    
    fig = px.bar(df, x=x_col, y=y_col, color=color_column, 
                facet_col=facet_column, hover_data=df.columns.tolist())
    
    if df[x_col].dtype == 'object':
        max_label_length = max(len(str(label)) for label in df[x_col])
        if max_label_length > 10:
            fig.update_xaxes(tickangle=45)
    
    return fig