# plik: data_loader.py
import torch
import numpy as np
import json
import sys
from ncmcm.bundlenet.bundlenet import BunDLeNet

def load_all_data(worm_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_custom_palette = [
    '#FF82AB',  
    '#8B4789',  
    '#87CEFA',  
    '#7CCD7C',  
    '#B3EE3A',  
    '#FF6A6A',  
    '#009ACD',  
    '#FFB90F'   
]

    # 1. Nazwy zachowań i neuronów
    with open("precomputed_data/behavior_names.json", "r") as f:
        behavior_names = {int(k): v for k, v in json.load(f).items()}

    with open(f"precomputed_data/data/worm_{worm_idx}_neuron_names.json", "r") as f:
        neuron_names = json.load(f)
    neuron_array = np.array(neuron_names)

    # 2. Macierze
    x_ = np.load(f"precomputed_data/data/worm_{worm_idx}_x.npy")
    b_ = np.load(f"precomputed_data/data/worm_{worm_idx}_b.npy").flatten()
    shap_matrix = np.load(rf"xai_matrices\worm_{worm_idx}_shap.npy")
    ig_matrix = np.load(rf"xai_matrices\worm_{worm_idx}_ig.npy")

    # 3. Przestrzeń Latentna 3D
    model = BunDLeNet(latent_dim=3, num_behaviour=len(behavior_names), input_shape=x_.shape).to(device)
    model.load_state_dict(torch.load(f"precomputed_data/models/worm_{worm_idx}_model.pt"))
    model.eval()

    with torch.no_grad():
        x_flat = x_[:, 0, 0, :] if len(x_.shape) == 4 else (x_[:, 0, :] if len(x_.shape) == 3 else x_)
        x_tensor = torch.tensor(x_flat.reshape(-1, x_.shape[-1]), dtype=torch.float32).to(device)
        latent_Y = model.tau(x_tensor).detach().cpu().numpy()

    return x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, my_custom_palette, model