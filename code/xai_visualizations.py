
# Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import shap
import torch
import plotly.graph_objects as go
import plotly.colors as pc
import ipywidgets as widgets
from plotly.subplots import make_subplots

################### GLOBAL PANEL ##########################################################################################

def plot_global_importance(attributions_shap, attributions_ig, b_labels, target_behavior=None, behavior_names=None,  neuron_names=None, top_n=10, title="Neuron Importance"):

    # Automaticly generating names for neurons (if not there)
    num_features = attributions_shap.shape[1]
    if neuron_names is None:
        neuron_names = [f"N {i}" for i in range(num_features)]

    #  Filltering thorught behaviour
    if target_behavior is not None:
        mask = (b_labels == target_behavior)
        
 
        if not np.any(mask):
            print(f"No bahaviour no. {target_behavior} in data.")
            return
            
        attr_shap_filtered = attributions_shap[mask]
        attr_ig_filtered = attributions_ig[mask]
        b_name = behavior_names[target_behavior] if behavior_names else str(target_behavior)
        title = f"{title} (Behaviour: {b_name})"
    else:
        attr_shap_filtered = attributions_shap
        attr_ig_filtered = attributions_ig
        title = f"{title} (Global)"

    mean_attr_shap = np.mean(np.abs(attr_shap_filtered), axis=0)
    mean_attr_ig = np.mean(np.abs(attr_ig_filtered), axis=0)  

    norm_shap = (mean_attr_shap - np.min(mean_attr_shap)) / (np.max(mean_attr_shap) - np.min(mean_attr_shap) + 1e-8)
    norm_ig = (mean_attr_ig - np.min(mean_attr_ig)) / (np.max(mean_attr_ig) - np.min(mean_attr_ig) + 1e-8)

    top_idx_shap = np.argsort(norm_shap)[-top_n:]
    top_idx_ig = np.argsort(norm_ig)[-top_n:]
    combined_idx = np.unique(np.concatenate([top_idx_shap, top_idx_ig]))

    ig_scores_subset = norm_ig[combined_idx]
    display_order = combined_idx[np.argsort(ig_scores_subset)[::-1]]

    top_neurons = np.array(neuron_names)[display_order]
    top_shap_vals = norm_shap[display_order]
    top_ig_vals = norm_ig[display_order]

    plt.figure(figsize=(12, 6))
    x_indexes = np.arange(len(top_neurons))
    width = 0.35 
    
    plt.bar(x_indexes - width/2, top_shap_vals, width=width, color="#8B0A50", label='SHAP')
    plt.bar(x_indexes + width/2, top_ig_vals, width=width, color="#00688B", label='Integrated Gradients')
    
    plt.title(title)
    plt.ylabel("Normlized attribuition")
    plt.xlabel("Neurons")
    
    plt.xticks(ticks=x_indexes, labels=top_neurons, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_comparison_table(attributions_shap, attributions_ig, b_labels, target_behavior=None, neuron_names=None, top_n=10):
   
    num_features = attributions_shap.shape[1]
    if neuron_names is None:
        neuron_names = [f"N {i}" for i in range(num_features)]
    neuron_array = np.array(neuron_names)

    
    if target_behavior is not None:
        mask = (b_labels == target_behavior)
        if not np.any(mask):
            print(f"No behaviour: {target_behavior} ")
            return None
        attr_shap = attributions_shap[mask]
        attr_ig = attributions_ig[mask]
    else:
        attr_shap = attributions_shap
        attr_ig = attributions_ig


    ig_scores = np.mean(np.abs(attr_ig), axis=0)
    shap_scores = np.mean(np.abs(attr_shap), axis=0)
    net_shap = np.mean(attr_shap, axis=0) 

    ig_idx = np.argsort(ig_scores)[-top_n:][::-1]
    shap_idx = np.argsort(shap_scores)[-top_n:][::-1]
    pos_idx = np.argsort(net_shap)[-top_n:][::-1]
    neg_idx = np.argsort(net_shap)[:top_n] 

    df = pd.DataFrame({
        'Place': range(1, top_n + 1),
        'Top IG': neuron_array[ig_idx],
        'Top SHAP (Absolutne)': neuron_array[shap_idx],
        'SHAP Positive (Netto +)': neuron_array[pos_idx],
        'SHAP Negative (Netto -)': neuron_array[neg_idx]
    })

    return df.set_index('Place')



def plot_shap_beeswarm(shap_attr, x_windowed, b_labels, target_behavior=None,behavior_names = None,  neuron_names=None):

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
            print(f"no behaviour: {target_behavior}.")
            return
        
        shap_vals = shap_attr[mask]
        feature_vals = x_flat[mask]
        b_name = behavior_names[target_behavior] if behavior_names else str(target_behavior)
        title = f"SHAP Beeswarm Plot (Behaviour: {b_name})"
    else:
        shap_vals = shap_attr
        feature_vals = x_flat
        title = "Global SHAP Beeswarm Plot"
        

    min_len = min(shap_vals.shape[0], feature_vals.shape[0])
    shap_vals = shap_vals[:min_len]
    feature_vals = feature_vals[:min_len]
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values=shap_vals, 
        features=feature_vals, 
        feature_names=list(neuron_names), 
        plot_type="dot",
        show=False 
    )
    
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()

####################### WORM SPECIFIC PANEL #########################################################


def create_wide_interactive_panel(x_, model, behaviors, shap_matrix, ig_matrix, behavior_names, neuron_names=None, palette=None):



    with torch.no_grad():
        num_features = x_.shape[-1]
    
        if len(x_.shape) == 4:
            x_flat = x_[:, 0, 0, :]
        elif len(x_.shape) == 3:
            x_flat = x_[:, 0, :]
        else:
            x_flat = x_

        x_tensor = torch.tensor(x_flat.reshape(-1, num_features), dtype=torch.float32)
        latent_Y = model.tau(x_tensor).cpu().numpy()


    num_features = shap_matrix.shape[1]
    if neuron_names is None:
        neuron_names = [f"N {i}" for i in range(num_features)]
    neuron_array = np.array(neuron_names)
    

    min_len = min(len(latent_Y), len(behaviors), len(shap_matrix), len(ig_matrix))
    latent_Y = latent_Y[:min_len]
    behaviors = behaviors[:min_len].flatten()  
    shap_matrix = shap_matrix[:min_len]
    ig_matrix = ig_matrix[:min_len]

    if palette is None:
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # ================= LEFT PANEL (3D) =================
    fig_3d = go.FigureWidget()
    
    fig_3d.add_trace(go.Scatter3d(
        x=latent_Y[:, 0], y=latent_Y[:, 1], z=latent_Y[:, 2],
        mode='lines',
        showlegend=False,
        line=dict(
            width=5, 
            color=behaviors,
            colorscale=palette[:len(behavior_names)], 
            cmin=0,             
            cmax=len(behavior_names)-1, 
            showscale=False 
        ),
        hoverinfo='text',
        text=[f"Time: {i}<br>Behaviour: {behavior_names[int(b)]}" for i, b in enumerate(behaviors)]
    ))
    

    for b_val in np.unique(behaviors):
        idx = int(b_val)
        color = palette[idx % len(palette)] 
        
        fig_3d.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color=color, width=5),
            name=behavior_names[idx],
            showlegend=True
        ))

    fig_3d.update_layout(
        title="Latent Space (Click on line)", 
        margin=dict(l=0, r=0, b=0, t=80), 
        width=600, height=800,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5)
    )

    # ================= RIGHT PANEL (XAI) =================
    fig_xai = go.FigureWidget(make_subplots(
        rows=2, cols=1, 
        subplot_titles=("SHAP attribuition", "IG attribuition"),
        vertical_spacing=0.25 
    ))
    
    fig_xai.add_trace(go.Bar(x=neuron_array, y=np.zeros(len(neuron_names)), marker_color='#8B0A50'), row=1, col=1)
    fig_xai.add_trace(go.Bar(x=neuron_array, y=np.zeros(len(neuron_names)), marker_color='#00688B'), row=2, col=1)
    
    fig_xai.update_xaxes(type='category', rangeslider=dict(visible=True, thickness=0.05))
    fig_xai.update_layout(showlegend=False, margin=dict(l=0, r=20, b=0, t=40), width=800, height=800)

    # ================= AKCJA PO KLIKNIĘCIU =================
    def update_xai(trace, points, state):
        if not points.point_inds: return
        t_idx = points.point_inds[0]
        
        # Pobranie danych dla danej klatki
        shap_raw = shap_matrix[t_idx]
        ig_raw = ig_matrix[t_idx]
        
        # Sortowanie po najważniejszych
        shap_idx = np.argsort(np.abs(shap_raw))[::-1]
        ig_idx = np.argsort(np.abs(ig_raw))[::-1]
        
        # Aktualizacja wykresów
        with fig_xai.batch_update():
            b_name = behavior_names[int(behaviors[t_idx])]
            fig_xai.layout.title.text = f"Time window: {t_idx} | Behaviour: {b_name}"
            
            fig_xai.data[0].x = neuron_array[shap_idx]
            fig_xai.data[0].y = shap_raw[shap_idx]
            
            fig_xai.data[1].x = neuron_array[ig_idx]
            fig_xai.data[1].y = ig_raw[ig_idx]
            
            fig_xai.layout.xaxis.range = [-0.5, 19.5]
            fig_xai.layout.xaxis2.range = [-0.5, 19.5]

    fig_3d.data[0].on_click(update_xai)

    layout = widgets.Layout(width='100%', display='flex', flex_flow='row')
    return widgets.HBox([fig_3d, fig_xai], layout=layout)

###### HEAT MAP PANEL ############################################################################

def create_interactive_temporal_dashboard(x_raw, behaviors, behavior_names, shap_matrix, ig_matrix, neuron_names, palette, top_n=10, title="Interactive Temporal Dashboard"):
    
    # 1. FIX DATA SHAPES 
    if len(x_raw.shape) == 4:
        if x_raw.shape[-1] > x_raw.shape[0]: 
            x_flat = x_raw[:, 0, 0, :].T 
        else:
            x_flat = x_raw[:, 0, 0, :]
    elif len(x_raw.shape) == 3:
        x_flat = x_raw[:, 0, :]
    else:
        x_flat = x_raw
        
    b_flat = behaviors.flatten()
    
    # Wyrównanie osi czasu dla wszystkich macierzy
    min_len = min(x_flat.shape[0], len(b_flat), shap_matrix.shape[0], ig_matrix.shape[0])
    x_flat = x_flat[:min_len]
    b_flat = b_flat[:min_len]
    shap_matrix = shap_matrix[:min_len]
    ig_matrix = ig_matrix[:min_len]
    
    num_neurons = x_flat.shape[1]
    
    # 2. GLOBAL NEURON SORTING (Kombinacja SHAP i IG)
    mean_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_ig = np.mean(np.abs(ig_matrix), axis=0)
    
    # Normalizujemy, by obie metody miały równy wpływ na sortowanie
    norm_shap = mean_shap / (np.max(mean_shap) + 1e-8)
    norm_ig = mean_ig / (np.max(mean_ig) + 1e-8)
    
    combined_mean = norm_shap + norm_ig
    sorted_idx = np.argsort(combined_mean)
    
    sorted_shap = shap_matrix[:, sorted_idx].T 
    sorted_ig = ig_matrix[:, sorted_idx].T 
    sorted_x_raw = x_flat[:, sorted_idx].T    
    sorted_names = np.array(neuron_names)[sorted_idx]
    
    # 3. PREPARE BEHAVIOR PALETTE & LEGEND
    num_classes = len(behavior_names)
    discrete_colorscale = []
    
    for i in range(num_classes):
        step_low = i / num_classes
        step_high = (i + 1) / num_classes
        c = palette[i % len(palette)]
        discrete_colorscale.append([step_low, c])
        discrete_colorscale.append([step_high, c])
        
    if isinstance(behavior_names, dict):
        b_texts = [f"Frame: {i}<br>Behavior: {behavior_names.get(int(b), str(b))}" for i, b in enumerate(b_flat)]
        ticktext = [behavior_names.get(i, str(i)) for i in range(num_classes)]
    else:
        b_texts = [f"Frame: {i}<br>Behavior: {behavior_names[int(b)]}" for i, b in enumerate(b_flat)]
        ticktext = [behavior_names[i] for i in range(num_classes)]
        
    tickvals = np.arange(num_classes) + 0.5
    
    # 4. CREATE PLOTLY DASHBOARD (4 Panele)
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True,           
        vertical_spacing=0.03,
        row_heights=[0.35, 0.05, 0.3, 0.3], # Zachowanie jest cienkie, reszta równa
        subplot_titles=("Raw Neuronal Activation", "Executed Behavior", "SHAP Influence", "Integrated Gradients Influence")
    )
    
    # --- PANEL 1: Raw Activation ---
    fig.add_trace(go.Heatmap(
        z=sorted_x_raw,
        y=sorted_names,
        colorscale='magma',
        colorbar=dict(title="Activation", x=1.01, y=0.84, len=0.3, yanchor="middle") 
    ), row=1, col=1)
    
    # --- PANEL 2: Behavior ---
    fig.add_trace(go.Heatmap(
        z=[b_flat],
        colorscale=discrete_colorscale,
        zmin=0,
        zmax=num_classes,
        showscale=True, 
        colorbar=dict(
            title="Behavior",
            tickvals=tickvals,
            ticktext=ticktext,
            x=1.01,
            y=0.63,
            len=0.1, # Mały suwaczek z etykietami
            yanchor="middle"
        ),
        hoverinfo="text",
        text=[b_texts]
    ), row=2, col=1)
    
    # --- PANEL 3: SHAP Heatmap ---
    max_shap = np.max(np.abs(sorted_shap))
    fig.add_trace(go.Heatmap(
        z=sorted_shap,
        y=sorted_names,
        colorscale='RdBu_r', 
        zmid=0,
        zmin=-max_shap,
        zmax=max_shap,
        colorbar=dict(title="SHAP Score", x=1.01, y=0.44, len=0.25, yanchor="middle") 
    ), row=3, col=1)
    
    # --- PANEL 4: IG Heatmap ---
    max_ig = np.max(np.abs(sorted_ig))
    fig.add_trace(go.Heatmap(
        z=sorted_ig,
        y=sorted_names,
        colorscale='RdBu_r', 
        zmid=0,
        zmin=-max_ig,
        zmax=max_ig,
        colorbar=dict(title="IG Score", x=1.01, y=0.14, len=0.25, yanchor="middle") 
    ), row=4, col=1)
    
    # 5. LAYOUT & INTERACTIVITY SETTINGS
    y_default_range = [num_neurons - top_n - 0.5, num_neurons - 0.5]
    
    fig.update_layout(
        title=title,
        height=1100,         # <- Zwiększona wysokość, by pomieścić 4 wykresy
        width=1400,          
        autosize=False,
        margin=dict(l=50, r=150, b=50, t=50), 
        hovermode="x unified" 
    )
    
    # Synchronizacja przewijania (matches='y') dla WSZYSTKICH macierzy
    fig.update_yaxes(range=y_default_range, title_text="Neurons", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1, fixedrange=True) 
    fig.update_yaxes(range=y_default_range, title_text="Neurons", matches='y', row=3, col=1)
    fig.update_yaxes(range=y_default_range, title_text="Neurons", matches='y', row=4, col=1)
    
    fig.update_xaxes(title_text="Time Frames", row=4, col=1)
    
    return go.FigureWidget(fig)