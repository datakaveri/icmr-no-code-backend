import plotly.express as px

def create_pie_plot(df, x_label, y_label, color_column, size_column, facet_column):
    names_col = df.columns[0]
    values_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    if len(df) > 10:
        df = df.head(10)
    
    if len(df.columns) == 1 or df[values_col].dtype == 'object':
        fig = px.pie(df, names=names_col, 
                    hover_data=df.columns.tolist())
    else:
        fig = px.pie(df, names=names_col, values=values_col, 
                    hover_data=df.columns.tolist())
    
    return fig