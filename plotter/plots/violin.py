import plotly.express as px

def create_violin_plot(df, x_label, y_label, color_column, size_column, facet_column):
    x_col = x_label if x_label and x_label in df.columns else df.columns[0]
    y_col = y_label if y_label and y_label in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
    
    if y_col:
        fig = px.violin(df, x=x_col, y=y_col, color=color_column, facet_col=facet_column,
                       hover_data=df.columns.tolist())
    else:
        fig = px.violin(df, y=x_col, color=color_column, facet_col=facet_column,
                       hover_data=df.columns.tolist())
    
    return fig