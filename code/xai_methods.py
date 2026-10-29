
# Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import shap

# Helper functions

class BehaviorWrapper(torch.nn.Module):
    """
    Wrpper for Bundle-net so we gat behavioural
    prediction right away, without the latent space
    """
    def __init__(self, bundle_model):
        super().__init__()
        self.model = bundle_model
        
    def forward(self, x):
        Y_t = self.model.tau(x)
        B_pred = self.model.predictor(Y_t)
        return B_pred

# Integrated gradient (IG)

def get_integrated_gradients(bundle_model, x_tensor, target_class):
    """
    Calculating integrated gradients for given dataset and behaviour

    Arguments:
    bundle_model = BunDLeNet(latent_dim, num_behaviour, input_shape)
    x_tensor = Neuronal data shaped as (t - win, 2, win, n) in a tensor format
    target_class = class of worms behaviour (inteagar from 1 to 8)

    """
    wrapper = BehaviorWrapper(bundle_model).eval()
    ig = IntegratedGradients(wrapper)
    mean_baseline = x_tensor.mean(dim=0, keepdim=True)
    attributions = ig.attribute(inputs=x_tensor, target=target_class, baselines=mean_baseline)
    
    return attributions.detach().cpu().numpy()


# SHAP 

def get_shap_values(bundle_model, x_tensor, target_class, background_tensor=None):
    wrapper = BehaviorWrapper(bundle_model).eval()
    
    if background_tensor is None:
        background_tensor = x_tensor
        
    explainer = shap.GradientExplainer(wrapper, background_tensor)
    shap_values = explainer.shap_values(x_tensor)
    shap_array = np.array(shap_values)
    batch_size = x_tensor.shape[0]

    if isinstance(shap_values, list):
        target_shap = shap_values[target_class]
    else:
        target_shap = shap_array[..., target_class]

    target_shap_2d = target_shap.reshape(batch_size, -1)
    
    return target_shap_2d


def generate_and_save_full_matrices(bundle_model, x_raw, b_raw, filename_prefix="worm_1"):

    num_features = x_raw.shape[-1]

    if len(x_raw.shape) == 4:
        x_raw = x_raw[:, 0, 0, :]
    elif len(x_raw.shape) == 3:
        x_raw = x_raw[:, 0, :]
        
    try:
        x_raw = x_raw.reshape(-1, num_features)
    except ValueError:
        raise ValueError(f"Cannot flatten. SHAPE: {x_raw.shape}")

    b_aligned = b_raw[:len(x_raw)]
    x_tensor = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor(b_aligned, dtype=torch.long)
    
    wrapper = BehaviorWrapper(bundle_model).eval()
    
    print(f"Calculating IG (num neurons: {num_features})...")
    ig = IntegratedGradients(wrapper)
    ig_attr = ig.attribute(x_tensor, target=targets, n_steps=20, internal_batch_size=256)
    ig_matrix = ig_attr.detach().cpu().numpy()
 
    print("Calculating SHAP...")
    explainer = shap.GradientExplainer(wrapper, x_tensor)
    shap_values = explainer.shap_values(x_tensor)
    
    shap_matrix = np.zeros_like(ig_matrix)
    shap_array = np.array(shap_values)

    for i in range(x_tensor.shape[0]):
        current_class = int(b_aligned[i])
        
        if isinstance(shap_values, list):
            frame_shap = shap_values[current_class][i]
        else:
            if shap_array.shape[-1] == num_features:
                frame_shap = shap_array[i, current_class, :]
            elif shap_array.shape[1] == num_features:
                frame_shap = shap_array[i, :, current_class]
            else:
                frame_shap = shap_array[i, ..., current_class]
                
        shap_matrix[i] = frame_shap.reshape(-1)
        

    np.save(f"{filename_prefix}_ig.npy", ig_matrix)
    np.save(f"{filename_prefix}_shap.npy", shap_matrix)
    print(f"Ready!  IG saved into ({ig_matrix.shape}) and SHAP into ({shap_matrix.shape})")
    
    return ig_matrix, shap_matrix