import json
import numpy as np
from pathlib import Path
# Zaimportuj swoje funkcje
from data_loader import load_all_data
from xai_tests import evaluate_rar, sperman_correlation_xai, inter_worm_correlation

def generate_quality_metrics(num_worms=5):
    all_metrics = {}

    for worm_idx in range(num_worms):
        print(f"Przetwarzanie robaka {worm_idx}...")
        x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx)
        
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
            table_title_all = "Brak danych"

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
                    'rar_shap': "Brak", 
                    'rar_ig': "Brak", 
                    'corr': "Brak",
                    'corr_p_value': "Brak",
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


    BASE_DIR = Path(__file__).resolve().parent
    output_path = BASE_DIR / 'precomputed_data' / 'quality_metrics.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"Saved into: {output_path}")



if __name__ == "__main__":
    generate_quality_metrics(num_worms=5)