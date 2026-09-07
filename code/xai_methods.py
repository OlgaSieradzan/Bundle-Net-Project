
# Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

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