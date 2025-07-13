import plotly.express as px

def create_histogram_plot(df, x_label, y_label, color_column, size_column, facet_column):
    x_col = x_label if x_label and x_label in df.columns else df.columns[0]
    
    fig = px.histogram(df, x=x_col, color=color_column, facet_col=facet_column,
                      hover_data=df.columns.tolist())
    
    return fig