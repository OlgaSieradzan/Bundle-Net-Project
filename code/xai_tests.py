
# Libraries
import torch
import numpy as np
from xai_methods import BehaviorWrapper

# Retain And Retrain (RAR) - Rahman et al. (2022)


def evaluate_rar(bundle_model, x_tensor, true_labels, attributions, retain_percentage=0.10):
    """
    Test for XAi validation - retain only top n% of neurons that were found most impactful,
    retrain the model and check how the accuracy changed (should stay almost the same)

    """
    # 1. Uśrednienie atrybucji dla każdego neuronu (po oknach i czasie)
    mean_attr = np.mean(np.abs(attributions), axis=(0, 1))
    
    # 2. Ustalenie progu ilościowego
    num_neurons = len(mean_attr)
    num_retain = max(1, int(num_neurons * retain_percentage))
    
    # 3. Wyłonienie indeksów najważniejszych neuronów
    top_indices = np.argsort(mean_attr)[-num_retain:]
    
    # 4. Generowanie binarnej maski (1 dla ważnych, 0 dla reszty)
    mask = torch.zeros(num_neurons, dtype=torch.float32)
    mask[top_indices] = 1.0
    
    # 5. Maskowanie sygnału na wejściu (wygaszenie nieważnych neuronów)
    masked_x = x_tensor * mask
    
    # 6. Ocena modelu na zredukowanych danych
    wrapper = BehaviorWrapper(bundle_model).eval()
    with torch.no_grad():
        predictions = wrapper(masked_x)
        predicted_classes = torch.argmax(predictions, dim=1)
        
        # Formatowanie etykiet referencyjnych
        if isinstance(true_labels, np.ndarray):
            true_labels = torch.tensor(true_labels, dtype=torch.long)
            
        correct = (predicted_classes == true_labels).sum().item()
        accuracy = correct / len(true_labels)
        
    return accuracy, num_retain