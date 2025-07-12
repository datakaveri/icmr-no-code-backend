import plotly.express as px

def create_heatmap_plot(df, x_label, y_label, color_column, size_column, facet_column):
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