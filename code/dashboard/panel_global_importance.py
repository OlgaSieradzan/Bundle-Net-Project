
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

def global_importance_barplot(shap_matrix, ig_matrix, b_labels, behavior_names, neuron_array, target_behavior, top_n=200):

    if target_behavior is not None:
            mask = (b_labels == target_behavior)
            if not np.any(mask):
                return go.Figure() 
            attr_shap = shap_matrix[mask]
            attr_ig = ig_matrix[mask]
            b_name = behavior_names.get(target_behavior, str(target_behavior)) if isinstance(behavior_names, dict) else behavior_names[target_behavior]
            title = f"Global Importance (Behaviour: {b_name})"
    else:
            attr_shap = shap_matrix
            attr_ig = ig_matrix
            title = "Globalne Importance (Whole time recording)"


    mean_shap = np.mean(np.abs(attr_shap), axis=0)
    mean_ig = np.mean(np.abs(attr_ig), axis=0)  

    norm_shap = (mean_shap - np.min(mean_shap)) / (np.max(mean_shap) - np.min(mean_shap) + 1e-8)
    norm_ig = (mean_ig - np.min(mean_ig)) / (np.max(mean_ig) - np.min(mean_ig) + 1e-8)

    combined_idx = np.unique(np.concatenate([np.argsort(norm_shap)[-top_n:], np.argsort(norm_ig)[-top_n:]]))
    display_order = combined_idx[np.argsort(norm_ig[combined_idx])[::-1]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=neuron_array[display_order], y=norm_shap[display_order], name='SHAP', marker_color='#8B0A50'))
    fig.add_trace(go.Bar(x=neuron_array[display_order], y=norm_ig[display_order], name='Integrated Gradients', marker_color='#00688B'))
    
    fig.update_layout(title= title, barmode='group', height=400, template="plotly_white")
    fig.update_xaxes(type='category', rangeslider=dict(visible=True, thickness=0.05), range=[-0.5, 19.5])
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
            return go.Figure() 
        attr_shap = attributions_shap[mask]
        attr_ig = attributions_ig[mask]
        b_name = behavior_names.get(target_behavior, str(target_behavior)) if isinstance(behavior_names, dict) else behavior_names[target_behavior]
        title = f"Comparassion of rankings (Behaviour: {b_name})"
    else:
        attr_shap = attributions_shap
        attr_ig = attributions_ig
        title = "Global Comparassion of rankings"

    ig_scores = np.mean(np.abs(attr_ig), axis=0)
    shap_scores = np.mean(np.abs(attr_shap), axis=0)
    net_shap = np.mean(attr_shap, axis=0) 

    actual_top_n = min(top_n, len(neuron_array))

    ig_idx = np.argsort(ig_scores)[-actual_top_n:][::-1]
    shap_idx = np.argsort(shap_scores)[-actual_top_n:][::-1]
    pos_idx = np.argsort(net_shap)[-actual_top_n:][::-1]
    neg_idx = np.argsort(net_shap)[:actual_top_n] 

    df = pd.DataFrame({
        'Place': range(1, len(ig_idx) + 1),
        'Top IG': neuron_array[ig_idx],
        'Top SHAP (Absolute)': neuron_array[shap_idx],
        'SHAP Positive (Netto +)': neuron_array[pos_idx],
        'SHAP Negative (Netto -)': neuron_array[neg_idx]
    })


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
    fig.update_layout(title=title, height=400, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
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
        title = f"SHAP Beeswarm Plot (Behaviour: {b_name})"
    else:
        shap_vals = shap_attr
        feature_vals = x_flat
        title = "Global SHAP Beeswarm Plot (Whole time recording)"

    min_len = min(shap_vals.shape[0], feature_vals.shape[0])
    shap_vals = shap_vals[:min_len]
    feature_vals = feature_vals[:min_len]

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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    plt.close(fig)

    encoded = base64.b64encode(buf.getbuffer()).decode("utf8")
    return f"data:image/png;base64,{encoded}"