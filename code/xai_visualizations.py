
# Libraries 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import shap
import torch

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


def plot_shap_beeswarm(shap_attr, x_raw, neuron_names):
    """
    Generuje wykres kropkowy SHAP. 
    Wymaga oryginalnych danych wejściowych (x_raw), aby przypisać kolory (poziom wapnia).
    """
    # Upewniamy się, że dane to płaskie tablice 2D (okna_czasowe, neurony)
    if torch.is_tensor(shap_attr):
        shap_attr = shap_attr.detach().cpu().numpy()
    if torch.is_tensor(x_raw):
        x_raw = x_raw.detach().cpu().numpy()
        
    # 2. Twarde wyrównanie wymiarów do 2D (czas, neurony)
    num_features = len(neuron_names)
    
    # -1 pozwala NumPy automatycznie wyliczyć poprawny czas
    shap_2d = shap_attr.reshape(-1, num_features)
    x_raw_2d = x_raw.reshape(-1, num_features)
    
    # 3. Zabezpieczenie przed niewłaściwą macierzą danych (obcinanie nadmiaru)
    if x_raw_2d.shape[0] != shap_2d.shape[0]:
        print(f"UWAGA: X ma {x_raw_2d.shape[0]} okien, a SHAP {shap_2d.shape[0]}. Wyrównuję.")
        x_raw_2d = x_raw_2d[:shap_2d.shape[0]]
        
    # 4. Rysowanie wykresu
    shap.summary_plot(
        shap_values=shap_2d, 
        features=x_raw_2d, 
        feature_names=list(neuron_names), 
        plot_type="dot",
        show=True
    )