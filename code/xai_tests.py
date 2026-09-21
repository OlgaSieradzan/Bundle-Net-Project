
# Libraries
import torch
import numpy as np
import json
import pandas as pd
from xai_methods import BehaviorWrapper
from scipy.stats import spearmanr
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space

# Retain And Retrain (RAR) - Rahman et al. (2022)


def evaluate_rar(x_windowed, b_labels, attributions, retain_percentage=0.10, epochs=200, device='cpu'):

    mean_attr = np.mean(np.abs(attributions), axis=0) 
    num_neurons = len(mean_attr)
    num_retain = max(1, int(num_neurons * retain_percentage))
    top_indices = np.argsort(mean_attr)[-num_retain:]

    x_retained = x_windowed[..., top_indices]

    num_behaviors = len(np.unique(b_labels))
    new_model = BunDLeNet(
        latent_dim=3, 
        num_behaviour=num_behaviors, 
        input_shape=x_retained.shape
    ).to(device)
    
    print(f"\n Traning (number of nurons: {num_retain})...")

    train_model(
        x_retained, b_labels, 
        new_model, 
        b_type='discrete', 
        gamma=0.9, 
        learning_rate=0.001, 
        n_epochs=epochs
    )
    new_model.eval()
    

    if len(x_retained.shape) == 4:
        x_flat = x_retained[:, 0, 0, :]
    elif len(x_retained.shape) == 3:
        x_flat = x_retained[:, 0, :]
    else:
        x_flat = x_retained
        
    x_flat = x_flat.reshape(-1, num_retain)
    x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(device)

    wrapper = BehaviorWrapper(new_model).eval()
    with torch.no_grad():
        predictions = wrapper(x_tensor)
        predicted_classes = torch.argmax(predictions, dim=1)
        true_labels_tensor = torch.tensor(b_labels, dtype=torch.long).to(device)
            
        correct = (predicted_classes == true_labels_tensor).sum().item()
        accuracy = correct / len(true_labels_tensor)
        
    return accuracy, num_retain

# XAi ranking correlation

def sperman_correlation_xai (shap_matrix, ig_matrix):

    mean_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_ig = np.mean(np.abs(ig_matrix), axis=0)

    correlation, p_value = spearmanr(mean_shap, mean_ig)

    return correlation, p_value

# Across worms correlation 

def get_worm_importance_dict(worm_idx, target_behavior, method='shap'):
    """Pobiera dane dla pojedynczego robaka i zwraca słownik: {nazwa_neuronu: średnia_ważność}"""
    try:
        b_ = np.load(f"precomputed_data/data/worm_{worm_idx}_b.npy")
        with open(f"precomputed_data/data/worm_{worm_idx}_neuron_names.json", "r") as f:
            neuron_names = json.load(f)

        if method.lower() == 'shap':
            attr = np.load(rf"xai_matrices\worm_{worm_idx}_shap.npy")
        else:
            attr = np.load(rf"xai_matrices\worm_{worm_idx}_ig.npy")
    except FileNotFoundError:
        return None  
        
    mask = (b_ == target_behavior)
    if not np.any(mask):
        return None  
        
    attr_filtered = attr[mask]

    mean_attr = np.mean(np.abs(attr_filtered), axis=0)

    return dict(zip(neuron_names, mean_attr))


def generate_inter_worm_correlation(target_behavior, ref_worm_idx=0, total_worms=5, method='shap', behavior_names=None):

    
    ref_data = get_worm_importance_dict(ref_worm_idx, target_behavior, method)
    b_name = behavior_names[target_behavior] if behavior_names else str(target_behavior)
    
    if not ref_data:
        print(f"No data for base worm (Worm {ref_worm_idx}) or no data on target behaviour: {b_name}")
        return None
        
    results = []
    
    for w_idx in range(total_worms):
        if w_idx == ref_worm_idx:
            continue
            
        other_data = get_worm_importance_dict(w_idx, target_behavior, method)
        
        if not other_data:
            results.append({"Comparrasion": f"Worm {w_idx}", "Common neurons": 0, "Correlation": np.nan, "P-value": "No data"})
            continue
            
        shared_neurons = set(ref_data.keys()).intersection(set(other_data.keys()))
        
        if len(shared_neurons) < 5:
            results.append({"Comparrasion": f"Worm {w_idx}", "Common neurons": len(shared_neurons), "Correlation": np.nan, "P-value": "Less then 5 shared neurons"})
            continue
            
        shared_neurons = sorted(list(shared_neurons))
        
        ref_vals = [ref_data[n] for n in shared_neurons]
        other_vals = [other_data[n] for n in shared_neurons]
        
        corr, p_val = spearmanr(ref_vals, other_vals)
        
        # ==========================================
        # ZMIANA: Przejrzyste formatowanie p-value
        # ==========================================
        if p_val < 0.05:
            p_str = "< 0.05"  # Znakomity wynik, pokazujemy od razu że jest istotny statystycznie
        else:
            p_str = f"{p_val:.3f}" # Brak istotności, pokazujemy 3 miejsca po przecinku
            
        results.append({
            "Comparrasion": f"Worm {w_idx}",
            "Common neurons": len(shared_neurons),
            "Correlation": round(corr, 3),
            "P-value": p_str
        })
        
    df = pd.DataFrame(results)
    df.index.name = f"{method.upper()} | {b_name} | Base: Worm {ref_worm_idx}"
    
    return df