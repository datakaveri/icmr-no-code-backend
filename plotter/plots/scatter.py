import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_scatter_plot(df, x_label, y_label, color_column, size_column, facet_column):
    if 'Cluster' in df.columns or 'cluster_id' in df.columns:
        exclude_cols = ['Cluster', 'cluster_id', 'point_id', 'Size', 'Distinctness']
        numeric_cols = [col for col in df.columns 
                      if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) >= 2:
            feature_data = df[numeric_cols].values
            
            scaler = StandardScaler()
            feature_data_scaled = scaler.fit_transform(feature_data)
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(feature_data_scaled)
            
            pca_df = df.copy()
            pca_df['PC1'] = pca_result[:, 0]
            pca_df['PC2'] = pca_result[:, 1]
            
            if 'cluster_id' in df.columns:
                color_col = 'cluster_id'
                color_title = 'Cluster ID'
            else:
                color_col = 'Cluster'
                color_title = 'Cluster'
            
            hover_data = {col: ':.3f' for col in numeric_cols}
            hover_data.update({
                'PC1': ':.3f',
                'PC2': ':.3f',
                color_col: True
            })
            
            # Define distinct colors for better visualization
            distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                             '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
            
            fig = px.scatter(
                pca_df, 
                x='PC1', 
                y='PC2',
                color=color_col,
                title=f"Clustering Analysis Results<br>" +
                      f"<sup>Features: {', '.join(numeric_cols[:8])}" +
                      f"{'...' if len(numeric_cols) > 8 else ''} " +
                      f"(Explained Variance: PC1={pca.explained_variance_ratio_[0]:.1%}, " +
                      f"PC2={pca.explained_variance_ratio_[1]:.1%})</sup>",
                labels={
                    'PC1': f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)',
                    'PC2': f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)',
                    color_col: color_title
                },
                hover_data=hover_data,
                color_discrete_sequence=distinct_colors
            )
            
            if 'cluster_id' in df.columns:
                cluster_centers_original = df.groupby('cluster_id')[numeric_cols].mean()
                
                centers_scaled = scaler.transform(cluster_centers_original.values)
                centers_pca = pca.transform(centers_scaled)
                
                for i, (cluster_id, center) in enumerate(zip(cluster_centers_original.index, centers_pca)):
                    hover_text = f"<b>Cluster {cluster_id} Center</b><br>"
                    hover_text += f"PC1: {center[0]:.3f}<br>"
                    hover_text += f"PC2: {center[1]:.3f}<br>"
                    hover_text += "<br><b>Original Feature Means:</b><br>"
                    for feature in numeric_cols:
                        hover_text += f"{feature}: {cluster_centers_original.loc[cluster_id, feature]:.3f}<br>"
                    
                    fig.add_trace(go.Scatter(
                        x=[center[0]], 
                        y=[center[1]],
                        mode='markers',
                        marker=dict(
                            size=15, 
                            color='black', 
                            symbol='diamond',
                            line=dict(width=2, color='white')
                        ),
                        name=f'Center {int(cluster_id)}',
                        showlegend=True,
                        hovertemplate=hover_text + "<extra></extra>"
                    ))
            
            # Removed the PC contributors annotation to prevent blocking the plot
            
            # Update layout for better appearance
            fig.update_layout(
                plot_bgcolor='rgba(40,40,40,1)',
                paper_bgcolor='rgba(40,40,40,1)',
                font=dict(color='white', size=12),
                title=dict(
                    font=dict(size=16, color='white'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    gridwidth=0.5,
                    zeroline=True,
                    zerolinecolor='rgba(128,128,128,0.5)',
                    zerolinewidth=1,
                    showline=False,
                    tickfont=dict(color='white'),
                    titlefont=dict(color='white')
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    gridwidth=0.5,
                    zeroline=True,
                    zerolinecolor='rgba(128,128,128,0.5)',
                    zerolinewidth=1,
                    showline=False,
                    tickfont=dict(color='white'),
                    titlefont=dict(color='white')
                ),
                legend=dict(
                    bgcolor="rgba(50,50,50,0.8)",
                    bordercolor="white",
                    borderwidth=1,
                    font=dict(color='white')
                ),
                margin=dict(l=80, r=80, t=100, b=80),
                width=900,
                height=650
            )
            
            # Remove the colorbar/gradient legend completely
            fig.update_coloraxes(showscale=False)
            
            return fig
        
        elif len(numeric_cols) == 1:
            x_col = numeric_cols[0]
            
            if 'cluster_id' in df.columns:
                color_col = 'cluster_id'
                color_title = 'Cluster ID'
            else:
                color_col = 'Cluster'
                color_title = 'Cluster'
            
            fig = px.scatter(
                df, 
                x=x_col, 
                y=[0] * len(df),
                color=color_col,
                title=f"1D K-Means Clustering: {x_col}",
                labels={'y': 'Cluster Distribution', color_col: color_title},
                hover_data=[col for col in df.columns if col != x_col],
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            y_jitter = np.random.normal(0, 0.1, len(df))
            fig.data[0].y = y_jitter
            
            if 'cluster_id' in df.columns:
                centers = df.groupby('cluster_id')[x_col].mean()
                for cluster_id, center_val in centers.items():
                    fig.add_vline(
                        x=center_val,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text=f"Center {cluster_id}: {center_val:.3f}",
                        annotation_position="top"
                    )
            
            fig.update_yaxes(range=[-0.5, 0.5], title="Cluster Distribution")
            
            return fig
        
        else:
            if 'Size' in df.columns:
                cluster_summary = df.groupby('Cluster')['Size'].first().reset_index()
                fig = px.bar(cluster_summary, x='Cluster', y='Size',
                           title="Cluster Sizes",
                           color='Cluster')
                return fig
    
    x_col = x_label if x_label and x_label in df.columns else df.columns[0]
    y_col = y_label if y_label and y_label in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    
    fig = px.scatter(df, x=x_col, y=y_col, color=color_column, size=size_column,
                    facet_col=facet_column, hover_data=df.columns.tolist())
    
    return fig