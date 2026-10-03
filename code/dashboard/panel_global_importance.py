
# Libraries

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64

# Code for visualizations

def global_importance_barplot(shap_matrix, ig_matrix, neuron_array, top_n=15):
    mean_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_ig = np.mean(np.abs(ig_matrix), axis=0)  

    norm_shap = (mean_shap - np.min(mean_shap)) / (np.max(mean_shap) - np.min(mean_shap) + 1e-8)
    norm_ig = (mean_ig - np.min(mean_ig)) / (np.max(mean_ig) - np.min(mean_ig) + 1e-8)

    combined_idx = np.unique(np.concatenate([np.argsort(norm_shap)[-top_n:], np.argsort(norm_ig)[-top_n:]]))
    display_order = combined_idx[np.argsort(norm_ig[combined_idx])[::-1]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=neuron_array[display_order], y=norm_shap[display_order], name='SHAP', marker_color='#8B0A50'))
    fig.add_trace(go.Bar(x=neuron_array[display_order], y=norm_ig[display_order], name='Integrated Gradients', marker_color='#00688B'))
    
    fig.update_layout(title="Global Neuron Importance", barmode='group', height=600, template="plotly_white")
    return fig

def comparison_table_xai(attributions_shap, attributions_ig, b_labels, behavior_names, neuron_names, target_behavior=None, top_n=10):
    """Zwraca tabelę rankingową jako interaktywny obiekt Plotly Table"""
    num_features = attributions_shap.shape[1]
    if neuron_names is None:
        neuron_names = [f"N {i}" for i in range(num_features)]
    neuron_array = np.array(neuron_names)
    
    if target_behavior is not None:
        mask = (b_labels == target_behavior)
        if not np.any(mask):
            return go.Figure() # Pusty wykres w przypadku braku zachowania
        attr_shap = attributions_shap[mask]
        attr_ig = attributions_ig[mask]
        b_name = behavior_names.get(target_behavior, str(target_behavior)) if isinstance(behavior_names, dict) else behavior_names[target_behavior]
        title = f"Porównanie Rankingów (Zachowanie: {b_name})"
    else:
        attr_shap = attributions_shap
        attr_ig = attributions_ig
        title = "Globalne Porównanie Rankingów Neuronów"

    ig_scores = np.mean(np.abs(attr_ig), axis=0)
    shap_scores = np.mean(np.abs(attr_shap), axis=0)
    net_shap = np.mean(attr_shap, axis=0) 

    ig_idx = np.argsort(ig_scores)[-top_n:][::-1]
    shap_idx = np.argsort(shap_scores)[-top_n:][::-1]
    pos_idx = np.argsort(net_shap)[-top_n:][::-1]
    neg_idx = np.argsort(net_shap)[:top_n] 

    df = pd.DataFrame({
        'Miejsce': range(1, top_n + 1),
        'Top IG': neuron_array[ig_idx],
        'Top SHAP (Absolutne)': neuron_array[shap_idx],
        'SHAP Positive (Netto +)': neuron_array[pos_idx],
        'SHAP Negative (Netto -)': neuron_array[neg_idx]
    })

    # Konwersja DataFrame do go.Table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='#f4f4f4',
                    align='center',
                    font=dict(size=14, color='black')),
        cells=dict(values=[df[col] for col in df.columns],
                   fill_color='white',
                   align='center',
                   font=dict(size=12, color='black'),
                   height=30))
    ])
    fig.update_layout(title=title, height=450, margin=dict(l=20, r=20, t=50, b=20), template="plotly_white")
    return fig

def shap_beeswarm(shap_attr, x_windowed, b_labels, behavior_names, neuron_names, target_behavior=None):
    """Generuje wykres SHAP Beeswarm i zwraca go jako zakodowany obraz Base64 dla Dash HTML"""
    num_features = shap_attr.shape[1]
    if neuron_names is None:
        neuron_names = [f"N {i}" for i in range(num_features)]

    if len(x_windowed.shape) == 4:
        x_flat = x_windowed[:, 0, 0, :]
    elif len(x_windowed.shape) == 3:
        x_flat = x_windowed[:, 0, :]
    else:
        x_flat = x_windowed

    if target_behavior is not None:
        mask = (b_labels == target_behavior)
        if not np.any(mask):
            return ""
        shap_vals = shap_attr[mask]
        feature_vals = x_flat[mask]
        
        b_name = behavior_names.get(target_behavior, str(target_behavior)) if isinstance(behavior_names, dict) else behavior_names[target_behavior]
        title = f"SHAP Beeswarm Plot (Zachowanie: {b_name})"
    else:
        shap_vals = shap_attr
        feature_vals = x_flat
        title = "Global SHAP Beeswarm Plot (Całe nagranie)"

    min_len = min(shap_vals.shape[0], feature_vals.shape[0])
    shap_vals = shap_vals[:min_len]
    feature_vals = feature_vals[:min_len]

    # Ustawienie backendu, żeby Matplotlib nie próbował otwierać okienek na serwerze
    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(12, 7))

    shap.summary_plot(
        shap_values=shap_vals, 
        features=feature_vals, 
        feature_names=list(neuron_names), 
        plot_type="dot",
        show=False 
    )
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    # Zapis obrazu do wirtualnego bufora w pamięci RAM
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # Kodowanie do HTML
    encoded = base64.b64encode(buf.getbuffer()).decode("utf8")
    return f"data:image/png;base64,{encoded}"