import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import os

def create_network_plot(df, x_label, y_label, color_column, size_column, facet_column):
    G = nx.DiGraph()
    
    csv_filename = 'association_rules.csv'
    if os.path.exists(csv_filename):
        csv_df = pd.read_csv(csv_filename)
        
        for _, row in csv_df.iterrows():
            G.add_edge(
                row['source'], 
                row['target'],
                weight=row['weight'],
                support=row.get('support', 0),
                lift=row.get('lift', 1),
                label=row['label']
            )
    else:
        if 'source' in df.columns and 'target' in df.columns:
            for _, row in df.iterrows():
                source = row['source']
                target = row['target']
                weight = row.get('weight', 1)
                label = row.get('label', f"{weight:.0%}")
                
                G.add_edge(source, target, weight=weight, label=label)
        elif len(df.columns) >= 2:
            source_col = df.columns[0]
            target_col = df.columns[1]
            weight_col = df.columns[2] if len(df.columns) > 2 else None
            
            for _, row in df.iterrows():
                source = row[source_col]
                target = row[target_col]
                weight = row[weight_col] if weight_col else 1
                label = f"{weight:.0%}" if weight_col else "100%"
                
                G.add_edge(source, target, weight=weight, label=label)
        else:
            raise ValueError("Network plot requires at least 2 columns (source, target)")
    
    node_centrality = nx.degree_centrality(G)
    
    pos = nx.spring_layout(G, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Centrality: {node_centrality[node]:.2f}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            size=20,
            color=list(node_centrality.values()),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Centrality")
        ),
        hoverinfo='text'
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Symptom Association Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
    )
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2,
            text=edge[2]['label'],
            showarrow=False,
            font=dict(size=10)
        )
    
    return fig