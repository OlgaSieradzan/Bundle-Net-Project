
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

def plot_global_importance(attributions_shap, attributions_ig, neuron_names, top_n=10, title="Global Neuron Importance"):

    num_features = len(neuron_names)

    ############ COVERSION OF DATA ##############################################
    if hasattr(attributions_shap, 'detach'):
        attributions_shap = attributions_shap.detach().cpu().numpy()
    if hasattr(attributions_ig, 'detach'):
        attributions_ig = attributions_ig.detach().cpu().numpy()
        
    try:
        attr_2d_shap = attributions_shap.reshape(-1, num_features)
        attr_2d_ig = attributions_ig.reshape(-1, num_features)
    except ValueError:
        raise ValueError("Error! shape doesnt match the neurons amout")
        
    ############# MEAN CALCULATION FROM TIME WINDOWS ############################
    mean_attr_shap = np.mean(np.abs(attr_2d_shap), axis=0)
    mean_attr_ig = np.mean(np.abs(attr_2d_ig), axis=0)  

    ############### MIN _MAX NORMALIZATION FOR COMPARASION######################
    norm_shap = (mean_attr_shap - np.min(mean_attr_shap)) / (np.max(mean_attr_shap) - np.min(mean_attr_shap))
    norm_ig = (mean_attr_ig - np.min(mean_attr_ig)) / (np.max(mean_attr_ig) - np.min(mean_attr_ig))


    ############### SELECTING TOP NEURONS ########################################
    top_idx_shap = np.argsort(norm_shap)[-top_n:]
    top_idx_ig = np.argsort(norm_ig)[-top_n:]
    combined_idx = np.unique(np.concatenate([top_idx_shap, top_idx_ig]))

    ig_scores_subset = norm_ig[combined_idx]
    display_order = combined_idx[np.argsort(ig_scores_subset)[::-1]]

    top_neurons = np.array(neuron_names)[display_order]
    top_shap_vals = norm_shap[display_order]
    top_ig_vals = norm_ig[display_order]

    ################ PLOT ########################################################
    plt.figure(figsize=(12, 6))
    
    x_indexes = np.arange(len(top_neurons))
    width = 0.35 
    
    plt.bar(x_indexes - width/2, top_shap_vals, width=width, color="#8B0A50", label='SHAP')
    plt.bar(x_indexes + width/2, top_ig_vals, width=width, color="#00688B", label='Integrated Gradients')
    
    plt.title(title)
    plt.ylabel("Normalized Importance (0 to 1)")
    plt.xlabel("Neuron")
    
    plt.xticks(ticks=x_indexes, labels=top_neurons, rotation=45)

    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_comparison_table(attributions_shap, attributions_ig, neuron_names, top_n=10):
    num_features = len(neuron_names)

    ############# PREPARING DATA #############################################
    if hasattr(attributions_shap, 'detach'):
        attributions_shap = attributions_shap.detach().cpu().numpy()
    if hasattr(attributions_ig, 'detach'):
        attributions_ig = attributions_ig.detach().cpu().numpy()
        
    attr_2d_shap = attributions_shap.reshape(-1, num_features)
    attr_2d_ig = attributions_ig.reshape(-1, num_features)

    ################ MEANS OF XAI ATTRIBUITION ###############################
    ig_scores = np.mean(np.abs(attr_2d_ig), axis=0)
    shap_scores = np.mean(np.abs(attr_2d_shap), axis=0)

    net_shap = np.mean(attr_2d_shap, axis=0) # SHap nettro is a sum of positive and negative influeances 

    ################# RANKING CRETAING ###########################
    ig_idx = np.argsort(ig_scores)[-top_n:][::-1]
    shap_idx = np.argsort(shap_scores)[-top_n:][::-1]
    pos_idx = np.argsort(net_shap)[-top_n:][::-1]
    neg_idx = np.argsort(net_shap)[:top_n]
    neuron_array = np.array(neuron_names)

    ############ DATA FRAME SHAPE ################################
    df = pd.DataFrame({
        'Place': range(1, top_n + 1),
        'Top IG': neuron_array[ig_idx],
        'Top SHAP (Global)': neuron_array[shap_idx],
        'SHAP Positive (Netto +)': neuron_array[pos_idx],
        'SHAP Negative (Netto -)': neuron_array[neg_idx]
    })

    return df.set_index('Place')



def plot_shap_beeswarm(shap_attr, x_raw, neuron_names):
    """
    Generuje wykres kropkowy SHAP. 
    Wymaga oryginalnych danych wejściowych (x_raw), aby przypisać kolory (poziom wapnia).
    """
    ############### DATA PREPARATION ##########################
    if torch.is_tensor(shap_attr):
        shap_attr = shap_attr.detach().cpu().numpy()
    if torch.is_tensor(x_raw):
        x_raw = x_raw.detach().cpu().numpy()
        
    num_features = len(neuron_names)
    shap_2d = shap_attr.reshape(-1, num_features)
    x_raw_2d = x_raw.reshape(-1, num_features)
    
    if x_raw_2d.shape[0] != shap_2d.shape[0]:
        print(f" Attencione!: X has a  {x_raw_2d.shape[0]} windowns, and SHAP has a {shap_2d.shape[0]}.")
        x_raw_2d = x_raw_2d[:shap_2d.shape[0]]
        
    ############## PLOTTING #####################################
    shap.summary_plot(
        shap_values=shap_2d, 
        features=x_raw_2d, 
        feature_names=list(neuron_names), 
        plot_type="dot",
        show=True
    )

####################### WORM SPECIFIC PANEL #########################################################


def create_wide_interactive_panel(latent_Y, behaviors, shap_matrix, ig_matrix, neuron_names, behavior_names, palette):
    neuron_array = np.array(neuron_names)
    
    # 1. Lewy panel: Model 3D
    fig_3d = go.FigureWidget()
    
    # Główna trajektoria używa teraz Twojej palety
    fig_3d.add_trace(go.Scatter3d(
        x=latent_Y[:, 0], y=latent_Y[:, 1], z=latent_Y[:, 2],
        mode='lines',
        line=dict(
            width=5, 
            color=behaviors,
            colorscale=palette, # Wstrzyknięcie własnych kolorów
            cmin=0,             # Twarde zakotwiczenie od pierwszego koloru
            cmax=len(palette)-1, # Twarde zakotwiczenie do ostatniego koloru
            showscale=False 
        ),
        hoverinfo='text',
        text=[f"Czas: {i}<br>Klasa: {behavior_names[int(b)]}" for i, b in enumerate(behaviors)]
    ))
    
    # Generowanie legendy w oparciu o Twoją paletę
    for b_val in np.unique(behaviors):
        idx = int(b_val)
        # Pobieramy kolor przypisany do konkretnego indeksu zachowania
        color = palette[idx % len(palette)] 
        
        fig_3d.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color=color, width=5),
            name=behavior_names[idx],
            showlegend=True
        ))

    fig_3d.update_layout(
        title="Latent Space (Kliknij linię)", 
        margin=dict(l=0, r=0, b=0, t=80), 
        width=600, height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    # 2. Prawy panel: Wykresy XAI (bez zmian)
    fig_xai = go.FigureWidget(make_subplots(
        rows=2, cols=1, 
        subplot_titles=("SHAP (Wartości Surowe)", "Integrated Gradients (Wartości Surowe)"),
        vertical_spacing=0.25 
    ))
    
    fig_xai.add_trace(go.Bar(x=neuron_array, y=np.zeros(len(neuron_names)), marker_color='#8B0A50'), row=1, col=1)
    fig_xai.add_trace(go.Bar(x=neuron_array, y=np.zeros(len(neuron_names)), marker_color='#00688B'), row=2, col=1)
    
    fig_xai.update_xaxes(type='category', rangeslider=dict(visible=True, thickness=0.05))
    fig_xai.update_layout(showlegend=False, margin=dict(l=0, r=20, b=0, t=40), width=800, height=800)

    # 3. Akcja po kliknięciu (bez zmian)
    def update_xai(trace, points, state):
        if not points.point_inds: return
        t_idx = points.point_inds[0]
        
        shap_raw = shap_matrix[t_idx]
        ig_raw = ig_matrix[t_idx]
        
        shap_idx = np.argsort(np.abs(shap_raw))[::-1]
        ig_idx = np.argsort(np.abs(ig_raw))[::-1]
        
        with fig_xai.batch_update():
            b_name = behavior_names[int(behaviors[t_idx])]
            fig_xai.layout.title.text = f"Klatka: {t_idx} | Klasa zachowania: {b_name}"
            
            fig_xai.data[0].x = neuron_array[shap_idx]
            fig_xai.data[0].y = shap_raw[shap_idx]
            
            fig_xai.data[1].x = neuron_array[ig_idx]
            fig_xai.data[1].y = ig_raw[ig_idx]
            
            fig_xai.layout.xaxis.range = [-0.5, 19.5]
            fig_xai.layout.xaxis2.range = [-0.5, 19.5]

    fig_3d.data[0].on_click(update_xai)

    layout = widgets.Layout(width='100%', display='flex', flex_flow='row')
    return widgets.HBox([fig_3d, fig_xai], layout=layout)



































def plot_transition_heatmap(attributions_window, neuron_names, top_n=15, title="Transition Trigger Heatmap"):

    """
    Generuje mapę cieplną dla pojedynczego okna czasowego (np. przy zmianie zachowania).
    """

    mean_attr = np.mean(np.abs(attributions_window), axis=1) 
    sorted_idx = np.argsort(mean_attr)[-top_n:]
    
    filtered_attr = attributions_window[sorted_idx, :]
    filtered_names = np.array(neuron_names)[sorted_idx]
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(filtered_attr, aspect='auto', cmap='coolwarm', 
                    vmin=-np.max(np.abs(filtered_attr)), vmax=np.max(np.abs(filtered_attr)))
    
    plt.yticks(ticks=np.arange(top_n), labels=filtered_names)
    plt.xlabel("Time steps (frames within window)")
    plt.title(title)
    plt.colorbar(im, label="IG Attribution Score")
    plt.tight_layout()
    plt.show()




def plot_full_temporal_dashboard(x_raw, behaviors, behavior_names, ig_matrix, neuron_names, top_n=10):
    """
    Tworzy 3-panelowy wykres: (1) Surowa aktywność, (2) Zachowanie, (3) Heatmapa IG w czasie.
    Zakładamy, że x_raw ma kształt (czas, neurony), a ig_matrix (czas, neurony).
    """
    # 1. Filtrowanie Top N neuronów z macierzy IG
    mean_attr = np.mean(np.abs(ig_matrix), axis=0) # Średnia ważność w całym czasie
    top_idx = np.argsort(mean_attr)[-top_n:][::-1] # Indeksy malejąco
    
    filtered_ig = ig_matrix[:, top_idx].T # Transponujemy do (neurony, czas) pod imshow
    filtered_names = np.array(neuron_names)[top_idx]
    
    # Tworzenie figury z 3 sub-wykresami (współdzielona oś X)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 0.5, 2]})
    
    # --- PANEL 1: Surowa aktywacja (jak na oryginale) ---
    im1 = ax1.imshow(x_raw.T, aspect='auto', cmap='magma', interpolation='nearest')
    ax1.set_ylabel("Neuronal activation")
    fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.01)
    
    # --- PANEL 2: Pasek zachowania ---
    # Tworzymy dyskretną mapę kolorów dla zachowań
    cmap_b = plt.cm.get_cmap('tab20', len(behavior_names))
    im2 = ax2.imshow(behaviors.reshape(1, -1), aspect='auto', cmap=cmap_b, interpolation='nearest')
    ax2.set_ylabel("Behaviour")
    ax2.set_yticks([]) # Ukrywamy os Y, bo to tylko jeden pasek
    
    # Legenda do zachowań
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.01, ticks=range(len(behavior_names)))
    cbar2.ax.set_yticklabels(behavior_names)
    
    # --- PANEL 3: Heatmapa XAI w czasie ---
    # Używamy coolwarm i centrujemy na zerze (vmin i vmax symetryczne)
    max_val = np.max(np.abs(filtered_ig))
    im3 = ax3.imshow(filtered_ig, aspect='auto', cmap='coolwarm', 
                     vmin=-max_val, vmax=max_val, interpolation='nearest')
    
    ax3.set_ylabel("Top IG Neurons")
    ax3.set_xlabel("time $t$")
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels(filtered_names)
    fig.colorbar(im3, ax=ax3, fraction=0.02, pad=0.01, label="IG Score")
    
    plt.tight_layout()
    plt.show()


