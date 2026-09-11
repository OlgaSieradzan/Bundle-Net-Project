
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

def get_integrated_gradients(bundle_model, x_tensor, target_class, baseline_value = None):
    """
    Calculating integrated gradients for given dataset and behaviour

    Arguments:
    bundle_model = BunDLeNet(latent_dim, num_behaviour, input_shape)
    x_tensor = Neuronal data shaped as (t - win, 2, win, n) in a tensor format
    target_class = class of worms behaviour (inteagar from 1 to 8) {?}
    baseline_value = value that integrated gradients takes as a comparassion to asses where are increased neuron activity.
                     Normaly can be a zero, a mean neauron activity or a gaussian noise

    """
    wrapper = BehaviorWrapper(bundle_model).eval()
    ig = IntegratedGradients(wrapper)
    attributions = ig.attribute(inputs=x_tensor, target=target_class, baselines = baseline_value)
    
    return attributions.detach().cpu().numpy()


# SHAP 

def get_shap_values(bundle_model, x_tensor, target_class, background_tensor=None):
    wrapper = BehaviorWrapper(bundle_model).eval()
    
    if background_tensor is None:
        background_tensor = x_tensor
        
    explainer = shap.GradientExplainer(wrapper, background_tensor)
    shap_values = explainer.shap_values(x_tensor)
    
    # 1. Konwersja do bezpiecznej macierzy NumPy
    shap_array = np.array(shap_values)
    batch_size = x_tensor.shape[0]
    
    # 2. Wycięcie klasy z ostatniego wymiaru
    # Operator "..." oznacza: "weź wszystkie klatki i neurony, ale tylko klasę 'target_class' na końcu"
    if isinstance(shap_values, list):
        target_shap = shap_values[target_class]
    else:
        target_shap = shap_array[..., target_class]
        
    # 3. Twarde wymuszenie ostatecznego kształtu (Klatki, Neurony)
    # -1 sprawia, że NumPy samo domyśli się, że reszta to 131 neuronów
    target_shap_2d = target_shap.reshape(batch_size, -1)
    
    return target_shap_2d