
# Libraries
import torch
import numpy as np
import json
import pandas as pd
from xai_methods import BehaviorWrapper
from scipy.stats import spearmanr
from ncmcm.bundlenet.utils import prep_data
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model, project_into_latent_space
from data_loader import load_all_data

# Retain And Retrain (RAR) - Rahman et al. (2022)


def evaluate_rar(x_windowed, b_labels, attributions, retain_percentage=0.10, epochs=200, device='cpu'):

    mean_attr = np.mean(np.abs(attributions), axis=0) 
    num_neurons = len(mean_attr)
    num_retain = max(1, int(num_neurons * retain_percentage))
    top_indices = np.argsort(mean_attr)[-num_retain:]

    X_train, Y_train = prep_data(x_windowed, b_labels)

    X_retained = X_train[..., top_indices]

    num_behaviors = len(np.unique(b_labels))
    new_model = BunDLeNet(
        latent_dim=3, 
        num_behaviour=num_behaviors, 
        input_shape=X_retained.shape
    ).to(device)
    
    #print(f"\n Traning (number of nurons: {num_retain})...")

    train_model(
        X_retained, Y_train, 
        new_model, 
        b_type='discrete', 
        gamma=0.9, 
        learning_rate=0.001, 
        n_epochs=epochs
    )
    new_model.eval()
    

    x_eval = X_retained[:, 0]

    if isinstance(x_eval, np.ndarray):
        x_tensor = torch.tensor(x_eval, dtype=torch.float32).to(device)
    else:
        x_tensor = x_eval.clone().detach().to(device).float()

    wrapper = BehaviorWrapper(new_model).eval()
    with torch.no_grad():
        predictions = wrapper(x_tensor)
        predicted_classes = torch.argmax(predictions, dim=1)
        true_labels_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
            
        correct = (predicted_classes == true_labels_tensor).sum().item()
        accuracy = correct / len(true_labels_tensor)
        
    return accuracy


# XAi ranking correlation

def sperman_correlation_xai (shap_matrix, ig_matrix):

    mean_shap = np.mean(np.abs(shap_matrix), axis=0)
    mean_ig = np.mean(np.abs(ig_matrix), axis=0)

    correlation, p_value = spearmanr(mean_shap, mean_ig)

    return correlation, p_value

# Across worms correlation 

def get_worm_all_metrics_dicts(x_flat, shap_matrix, ig_matrix, b_labels, neuron_names, worm_idx, target_behavior):

    try:

        if target_behavior == 'ALL':
            mask = np.ones(len(b_labels), dtype=bool)
        else:
            mask = (b_labels == int(target_behavior))

        if not np.any(mask):
            return {}, {}, {}

        x_filtered = x_flat[mask]
        shap_filtered = shap_matrix[mask]
        ig_filtered = ig_matrix[mask]

        mean_activity = np.mean(np.abs(x_filtered), axis=0)
        mean_shap = np.mean(np.abs(shap_filtered), axis=0)
        mean_ig = np.mean(np.abs(ig_filtered), axis=0)

        activity_dict = dict(zip(neuron_names, mean_activity))
        shap_dict = dict(zip(neuron_names, mean_shap))
        ig_dict = dict(zip(neuron_names, mean_ig))

        return shap_dict, ig_dict, activity_dict

    except Exception as e:
        print(f" Error for worem {worm_idx}: {e}")
        return None, None, None

    
def get_corr_and_pval(ref_dict, other_dict, shared_neurons):
            r_vals = [ref_dict[n] for n in shared_neurons]
            o_vals = [other_dict[n] for n in shared_neurons]
            corr, p_val = spearmanr(r_vals, o_vals)
            p_str = "< 0.05" if p_val < 0.05 else f"{p_val:.3f}"
            return round(corr, 3), p_str


def inter_worm_correlation(target_behavior, ref_worm_idx=0, total_worms=5, behavior_names=None):

    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=ref_worm_idx)

    ref_shap, ref_ig, ref_act = get_worm_all_metrics_dicts(x_flat, shap_matrix, ig_matrix, b_, neuron_array, ref_worm_idx)
    
    b_name = behavior_names[target_behavior] if behavior_names else str(target_behavior)
    
    if not ref_shap:
        return None
        
    results = []
    
    for w_idx in range(total_worms):
        if w_idx == ref_worm_idx:
            continue
            
        x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=w_idx)
        
        other_shap, other_ig, other_act = get_worm_all_metrics_dicts(x_flat, shap_matrix, ig_matrix, b_, neuron_array, w_idx)
        
        if not other_shap:
            results.append({
                "Comparrasion": f"Worm {w_idx}", "Common neurons": 0, 
                "SHAP Corr": np.nan, "SHAP P-val": "No data",
                "IG Corr": np.nan, "IG P-val": "No data",
                "Activity Corr": np.nan, "Activity P-val": "No data"
            })
            continue

        shared_neurons = set(ref_shap.keys()).intersection(set(other_shap.keys()))
        
        if len(shared_neurons) < 5:
            results.append({
                "Comparrasion": f"Worm {w_idx}", "Common neurons": len(shared_neurons), 
                "SHAP Corr": np.nan, "SHAP P-val": "< 5 neurons",
                "IG Corr": np.nan, "IG P-val": "No data",
                "Activity Corr": np.nan, "Activity P-val": "No data"
                
            })
            continue
            
        shared_neurons = sorted(list(shared_neurons))
        shap_corr, shap_p = get_corr_and_pval(ref_shap, other_shap, shared_neurons)
        ig_corr, ig_p = get_corr_and_pval(ref_ig, other_ig), shared_neurons
        act_corr, act_p = get_corr_and_pval(ref_act, other_act, shared_neurons)
            
        results.append({
            "Comparrasion": f"Worm {w_idx}",
            "Common neurons": len(shared_neurons),
            "SHAP Corr": shap_corr, "SHAP P-val": shap_p,
            "IG Corr": ig_corr, "IG P-val": ig_p,
            "Activity Corr": act_corr, "Activity P-val": act_p
        })
        
    df = pd.DataFrame(results)
    df.index.name = f"Multi-Metric | {b_name} | Base: Worm {ref_worm_idx}"
    return df