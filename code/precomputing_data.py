import os
import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Importy z BunDLe-Net
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model

def prepare_and_save_all(num_worms, database_path):

    # Configurating the folder
    base_dir = "precomputed_data"
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    label_encoder = LabelEncoder()

    # main loop
    for worm_idx in range(num_worms):

        data = Database(data_path=database_path, dataset_no=worm_idx)
        print(f" Computing for worm: {worm_idx + 1}/{num_worms} ")

        behaviors_dict = {str(k): str(v) for k, v in data.behaviour_names.items()}
    
        with open(os.path.join(base_dir, "behavior_names.json"), "w") as f:
            json.dump(behaviors_dict, f)
    
        try:
            
            x_raw = data.neuron_traces.T
            b_raw = data.behaviour

            num_behaviors = len(data.behaviour_names)
            with open(os.path.join(base_dir, "num_behaviors.txt"), "w") as f:
                f.write(str(num_behaviors))

            n_names = [str(n) for n in data.neuron_names]
            with open(os.path.join(data_dir, f"worm_{worm_idx}_neuron_names.json"), "w") as f:
                json.dump(n_names, f)

            b_encoded = label_encoder.fit_transform(b_raw.ravel())
            x_, b_ = prep_data(x_raw, b_encoded, win=1)

            np.save(os.path.join(data_dir, f"worm_{worm_idx}_x.npy"), x_)
            np.save(os.path.join(data_dir, f"worm_{worm_idx}_b.npy"), b_)
            print(f"data saved: x_ {x_.shape}, b_ {b_.shape}")
 
            print("Model traning...")
            model = BunDLeNet(
                latent_dim=3, 
                num_behaviour=num_behaviors, 
                input_shape=x_.shape
            ).to(device)

            train_model(
                x_, b_, 
                model, 
                b_type='discrete', 
                gamma=0.9, 
                learning_rate=0.001, 
                n_epochs=200 
            )
            
            torch.save(model.state_dict(), os.path.join(model_dir, f"worm_{worm_idx}_model.pt"))
            print(f" model for worm {worm_idx} is saved.")
            
        except Exception as e:
            print(f"Error with worm: {worm_idx}: {e}")

if __name__ == "__main__":
    prepare_and_save_all(5, "C:/Users/olgas/GitHub/Bundle-Net-Project/dataset/c_elegans/NoStim_Data.mat")