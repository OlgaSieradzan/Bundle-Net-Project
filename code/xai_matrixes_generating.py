import os
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Importy z Twojego środowiska BunDLe-Net
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model

# Importy Twoich metod XAI (upewnij się, że funkcja jest w xai_methods.py)
from xai_methods import generate_and_save_full_matrices

def process_all_worms(num_worms):
    output_dir = "xai_matrices"
    os.makedirs(output_dir, exist_ok=True)

    for worm_idx in (num_worms):
        print(f"\n--- Przetwarzanie robaka {worm_idx + 1}/{num_worms} ---")
        
        try:

            # Database loding
            data_path = "C:/Users/olgas/GitHub/Bundle-Net-Project/dataset/c_elegans/NoStim_Data.mat"
            data = Database(data_path=data_path, dataset_no=worm_idx)
            x = data.neuron_traces.T
            b = data.behaviour

            # Transformation 
            label_encoder = LabelEncoder()
            b = label_encoder.fit_transform(b)
            print(x.shape, b.shape)
            x_, b_ = prep_data(x, b, win=1)
            print(x_.shape, b_.shape)

            # Model traning 
            model = BunDLeNet(latent_dim=3, num_behaviour=len(data.behaviour_names), input_shape=x_.shape)
            loss_array, _ = train_model(
                x_,
                b_,
                model,
                b_type='discrete',
                gamma=0.9,
                learning_rate=0.001,
                n_epochs=500
            )

            prefix = os.path.join(output_dir, f"worm_{worm_idx}")
            generate_and_save_full_matrices(
                bundle_model=model, 
                x_raw=x_, 
                b_raw=b_, 
                filename_prefix=prefix
            )
            
        except Exception as e:
            print(f"Błąd podczas przetwarzania robaka {worm_idx}: {e}")
            continue

if __name__ == "__main__":
    worms = [0,1,2,3,4]
    process_all_worms(worms)