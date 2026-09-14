
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


def generate_and_save_full_matrices(bundle_model, x_raw, b_raw, filename_prefix="worm_1"):
    # 1. NAPRAWA WYMIARÓW
    if len(x_raw.shape) > 2:
        x_raw = x_raw[:, 0]
        
    num_features = 131
    try:
        x_raw = x_raw.reshape(x_raw.shape[0], num_features)
    except ValueError:
        raise ValueError(f"Nie można spłaszczyć danych do 131 neuronów. Kształt: {x_raw.shape}")

    b_aligned = b_raw[:len(x_raw)]
    x_tensor = torch.tensor(x_raw, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor(b_aligned, dtype=torch.long)
    
    wrapper = BehaviorWrapper(bundle_model).eval()
    
    # 2. Generowanie Integrated Gradients
    print("Obliczam Integrated Gradients...")
    ig = IntegratedGradients(wrapper)
    ig_attr = ig.attribute(x_tensor, target=targets, n_steps=20, internal_batch_size=256)
    ig_matrix = ig_attr.detach().cpu().numpy()
    
    # 3. Generowanie SHAP
    print("Obliczam SHAP... (To potrwa, zrób sobie przerwę)")
    explainer = shap.GradientExplainer(wrapper, x_tensor)
    shap_values = explainer.shap_values(x_tensor)
    
    shap_matrix = np.zeros_like(ig_matrix)
    shap_array = np.array(shap_values)
    
    # 4. Twarde i bezpieczne przypisywanie klas (Kotwica na 131 neuronach)
    for i in range(x_tensor.shape[0]):
        current_class = int(b_aligned[i])
        
        if isinstance(shap_values, list):
            frame_shap = shap_values[current_class][i]
        else:
            # Sprawdzamy, w którym miejscu SHAP ukrył wymiar neuronów
            if shap_array.shape[-1] == num_features:
                # Układ np. (3058, 8, 131)
                frame_shap = shap_array[i, current_class, :]
            elif shap_array.shape[1] == num_features:
                # Układ np. (3058, 131, 8) - to przypadek z Twojego błędu!
                frame_shap = shap_array[i, :, current_class]
            else:
                # Fallback, jeśli wymiary znów zmutują
                frame_shap = shap_array[i, ..., current_class]
                
        shap_matrix[i] = frame_shap.reshape(-1)
        
    # 5. Zapis do plików numpy
    np.save(f"{filename_prefix}_ig.npy", ig_matrix)
    np.save(f"{filename_prefix}_shap.npy", shap_matrix)
    print(f"Gotowe! Zapisano IG ({ig_matrix.shape}) oraz SHAP ({shap_matrix.shape})")
    
    return ig_matrix, shap_matrix