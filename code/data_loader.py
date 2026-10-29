# plik: data_loader.py
import torch
import numpy as np
import json
import sys
import os
from ncmcm.bundlenet.bundlenet import BunDLeNet

def load_all_data(worm_idx, precomputed_data_path, xai_matrices_path):
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

    with open( os.path.join(precomputed_data_path, "behavior_names.json"), "r") as f:
                    behavior_names = {int(k): v for k, v in json.load(f).items()}

    with open( os.path.join(precomputed_data_path, f"data/worm_{worm_idx}_neuron_names.json") , "r") as f:
        neuron_names = json.load(f)
    neuron_array = np.array(neuron_names)


    x_ = np.load(os.path.join(precomputed_data_path, "data", f"worm_{worm_idx}_x.npy"))
    b_ = np.load(os.path.join(precomputed_data_path, "data", f"worm_{worm_idx}_b.npy")).flatten()

    shap_matrix = np.load(os.path.join(xai_matrices_path,f"worm_{worm_idx}_shap.npy"))
    ig_matrix = np.load(os.path.join(xai_matrices_path,f"worm_{worm_idx}_ig.npy"))

    model = BunDLeNet(latent_dim=3, num_behaviour=len(behavior_names), input_shape=x_.shape)
    model_path = os.path.join(precomputed_data_path, "models", f"worm_{worm_idx}_model.pt")
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        x_flat = x_[:, 0, 0, :] if len(x_.shape) == 4 else (x_[:, 0, :] if len(x_.shape) == 3 else x_)
        x_tensor = torch.tensor(x_flat.reshape(-1, x_.shape[-1]), dtype=torch.float32).to(device)
        latent_Y = model.tau(x_tensor).detach().cpu().numpy()

    return x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, my_custom_palette, model

METRICS_FILE = "C:/Users/olgas/GitHub/Bundle-Net-Project/code/precomputed_data/quality_metrics.json"
DATA_PATH = "C:/Users/olgas/GitHub/Bundle-Net-Project/code/precomputed_data"
XAI_PATH = "C:/Users/olgas/GitHub/Bundle-Net-Project/code/xai_matrices"