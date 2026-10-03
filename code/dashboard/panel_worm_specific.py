# Libraries

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch

# Code for visualizations

def latent_space_3D(x_, model, b_, behavior_names, palette):

    with torch.no_grad():
            num_features = x_.shape[-1]
        
            if len(x_.shape) == 4:
                x_flat = x_[:, 0, 0, :]
            elif len(x_.shape) == 3:
                x_flat = x_[:, 0, :]
            else:
                x_flat = x_
            device = next(model.parameters()).device
            x_tensor = torch.tensor(x_flat.reshape(-1, num_features), dtype=torch.float32).to(device)
            
            latent_Y = model.tau(x_tensor).cpu().numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=latent_Y[:, 0], y=latent_Y[:, 1], z=latent_Y[:, 2],
        mode='lines', showlegend=False,
        line=dict(width=5, color=b_, colorscale=palette[:len(behavior_names)], cmin=0, cmax=len(behavior_names)-1),
        hoverinfo='text',
        text=[f"Time: {i}<br>Behaviour: {behavior_names.get(int(b), str(b))}" for i, b in enumerate(b_)],
        customdata=np.arange(len(b_)) 
    ))
    for b_val in np.unique(b_):
        idx = int(b_val)
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines', line=dict(color=palette[idx % len(palette)], width=5), name=behavior_names.get(idx, str(idx))))
    fig.update_layout(title=dict(text="Latent Space (Click on the line)", y=0.98), margin=dict(l=0, r=0, b=100, t=80), height=700, legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5))
    return fig

def empty_xai_bar_plots():
    fig = make_subplots(rows=2, cols=1, subplot_titles=("SHAP attribuition", "IG attribuition"), vertical_spacing=0.25)
    fig.add_trace(go.Bar(x=[], y=[], marker_color='#8B0A50', name="SHAP"), row=1, col=1)
    fig.add_trace(go.Bar(x=[], y=[], marker_color='#00688B', name="IG"), row=2, col=1)
    fig.update_xaxes(type='category', rangeslider=dict(visible=True, thickness=0.05))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=20, b=0, t=40), height=700, template="plotly_white")
    return fig

def CLICK_xai_barplots(t_idx, shap_matrix, ig_matrix, neuron_array, b_, behavior_names):

    shap_raw, ig_raw = shap_matrix[t_idx], ig_matrix[t_idx]
    shap_idx = np.argsort(np.abs(shap_raw))[::-1]
    ig_idx = np.argsort(np.abs(ig_raw))[::-1]
    
    b_name = behavior_names.get(int(b_[t_idx]), str(int(b_[t_idx])))
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("SHAP attribuition", "IG attribuition"), vertical_spacing=0.25)
    fig.add_trace(go.Bar(x=neuron_array[shap_idx], y=shap_raw[shap_idx], marker_color='#8B0A50'), row=1, col=1)
    fig.add_trace(go.Bar(x=neuron_array[ig_idx], y=ig_raw[ig_idx], marker_color='#00688B'), row=2, col=1)
    
    fig.update_xaxes(type='category', rangeslider=dict(visible=True, thickness=0.05), range=[-0.5, 19.5])
    fig.update_layout(showlegend=False, margin=dict(l=0, r=20, b=0, t=60), height=700, title=f"Time window: {t_idx} | Behaviour: {b_name}", template="plotly_white")
    return fig