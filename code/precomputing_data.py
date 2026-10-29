import os
import json
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ncmcm.data_loaders.matlab_dataset import Database
from ncmcm.bundlenet.utils import prep_data
from ncmcm.bundlenet.bundlenet import BunDLeNet, train_model
from xai_methods import generate_and_save_full_matrices
from xai_tests import evaluate_rar, sperman_correlation_xai, inter_worm_correlation
from data_loader import load_all_data



def precompute_data_and_models(num_worms, database_path, output_folder, gamma_rate):

    # Configurating the folder
    base_dir = output_folder
    data_dir = os.path.join(base_dir, "data")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    label_encoder = LabelEncoder()

    b_neurons = [
                    'AVAR',
                    'AVAL',
                    'SMDVR',
                    'SMDVL',
                    'SMDDR',
                    'SMDDL',
                    'RIBR',
                    'RIBL'
                ]

    # main loop
    for worm_idx in range(num_worms):


        data = Database(data_path=database_path, dataset_no=worm_idx)
        data.exclude_neurons(b_neurons)
        print(f"Computing for worm: {worm_idx + 1}/{num_worms} ")

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
                gamma= gamma_rate, 
                learning_rate= 0.001, 
                n_epochs= 1000 
            )
            
            torch.save(model.state_dict(), os.path.join(model_dir, f"worm_{worm_idx}_model.pt"))
            print(f" model for worm {worm_idx} is saved.")
            
        except Exception as e:
            print(f"Error with worm: {worm_idx}: {e}")


def precompute_xai(num_worms, input_folder, output_folder):
    output_dir = output_folder
    os.makedirs(output_dir, exist_ok=True)

    for worm_idx in range(num_worms):
        print(f"\n--- Calculating for worm {worm_idx + 1}/{num_worms} ---")
        
        try:
            x_ = np.load(os.path.join(input_folder, "data", f"worm_{worm_idx}_x.npy"))
            b_ = np.load(os.path.join(input_folder, "data", f"worm_{worm_idx}_b.npy")).flatten()
            
            with open(os.path.join(input_folder, "behavior_names.json"), "r") as f:
                behavior_names = {int(k): v for k, v in json.load(f).items()}

            model = BunDLeNet(latent_dim=3, num_behaviour=len(behavior_names), input_shape=x_.shape)
            
            model_path = os.path.join(input_folder, "models", f"worm_{worm_idx}_model.pt")
            model.load_state_dict(torch.load(model_path))

            prefix = os.path.join(output_dir, f"worm_{worm_idx}")
            generate_and_save_full_matrices(
                bundle_model=model, 
                x_raw=x_, 
                b_raw=b_, 
                filename_prefix=prefix
            )
            
        except Exception as e:
            print(f"Error with worm: {worm_idx}: {e}")
            continue


def precompute_quality_metrics(num_worms, precomputed_data_path, xai_matrices_path):
    all_metrics = {}

    for worm_idx in range(num_worms):
        print(f" Computing for worm {worm_idx}...")
        x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx, precomputed_data_path, xai_matrices_path)
        
        all_metrics[str(worm_idx)] = {}
        rar_global_shap = evaluate_rar(x_flat, b_, shap_matrix) 
        rar_global_ig = evaluate_rar(x_flat, b_, ig_matrix)
        corr_global, p_value_global = sperman_correlation_xai(shap_matrix, ig_matrix) 

        df_multi_all = inter_worm_correlation(
            target_behavior='ALL', 
            ref_worm_idx=worm_idx, 
            behavior_names=behavior_names
        )
        
        if df_multi_all is not None:
            table_records_all = df_multi_all.to_dict('records') 
            table_title_all = df_multi_all.index.name
        else:
            table_records_all = []
            table_title_all = "No data"

        all_metrics[str(worm_idx)]['ALL'] = {
            'rar_shap': round(rar_global_shap, 3),
            'rar_ig': round(rar_global_ig, 3),
            'corr': round(corr_global, 3),
            'corr_p_value': round(p_value_global, 3),
            'inter_worm_table': {
                'title': table_title_all,
                'data': table_records_all
            }
        }

        for beh_idx, beh_name in behavior_names.items():
            mask = (b_ == int(beh_idx))

            if not np.any(mask):
                all_metrics[str(worm_idx)][str(beh_idx)] = {
                    'rar_shap': "no data", 
                    'rar_ig': "no data", 
                    'corr': "no data",
                    'corr_p_value': "no data",
                    'inter_worm_table': {'title': "No behaviour", 'data': []}
                }
                continue

            shap_behavior = shap_matrix[mask]
            ig_behavior = ig_matrix[mask]

            rar_beh_shap = evaluate_rar(x_flat, b_, shap_behavior) 
            rar_beh_ig = evaluate_rar(x_flat, b_, ig_behavior) 
            corr_beh, p_value_beh = sperman_correlation_xai(shap_behavior, ig_behavior)
            
            df_multi = inter_worm_correlation(
                target_behavior=beh_idx, 
                ref_worm_idx=worm_idx, 
                behavior_names=behavior_names
            )
            
            if df_multi is not None:
                table_records = df_multi.to_dict('records') 
                table_title = df_multi.index.name
            else:
                table_records = []
                table_title = "No data"
            
            all_metrics[str(worm_idx)][str(beh_idx)] = {
                'rar_shap': round(rar_beh_shap, 3),
                'rar_ig': round(rar_beh_ig, 3),
                'corr': round(corr_beh, 3),
                'corr_p_value': round(p_value_beh, 3),
                'inter_worm_table': {
                    'title': table_title,
                    'data': table_records
                }
            }

    
    with open(os.path.join(precomputed_data_path, "quality_metrics"), 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"Saved into: {precomputed_data_path}")

if __name__ == "__main__":

    # precompute_data_and_models(5, "C:/Users/olgas/GitHub/Bundle-Net-Project/dataset/c_elegans/NoStim_Data.mat", "precomputed_data_70", 0.7)
    # precompute_data_and_models(5, "C:/Users/olgas/GitHub/Bundle-Net-Project/dataset/c_elegans/NoStim_Data.mat", "precomputed_data_80", 0.8)
    # precompute_data_and_models(5, "C:/Users/olgas/GitHub/Bundle-Net-Project/dataset/c_elegans/NoStim_Data.mat", "precomputed_data_90", 0.9)
    # precompute_data_and_models(5, "C:/Users/olgas/GitHub/Bundle-Net-Project/dataset/c_elegans/NoStim_Data.mat", "precomputed_data_95", 0.95)

    precompute_xai(5, "precomputed_data_70", "xai_matrices_70")
    precompute_xai(5, "precomputed_data_80", "xai_matrices_80")
    precompute_xai(5, "precomputed_data_90", "xai_matrices_90")
    precompute_xai(5, "precomputed_data_95", "xai_matrices_95")

    precompute_quality_metrics(5, "precomputed_data_70", "xai_matrices_70")
    precompute_quality_metrics(5, "precomputed_data_70", "xai_matrices_80")
    precompute_quality_metrics(5, "precomputed_data_70", "xai_matrices_90")
    precompute_quality_metrics(5, "precomputed_data_70", "xai_matrices_95")

