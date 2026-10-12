
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_loader import load_all_data



def inter_worm_corr(target_behavior, target_metric, num_worms=5):
    
  # Pobieramy nazwy z pierwszego robaka jako bazę do tytułu
    _, _, _, _, behavior_names, _, _, _ = load_all_data(worm_idx=0)
    
    if target_behavior != "ALL":
        target_behavior = int(target_behavior)
        b_name = behavior_names.get(target_behavior, str(target_behavior))
    else:
        b_name = "ALL"

    worm_series_list = []

    for w in range(num_worms):
        # Pobieramy też neuron_array, żeby wiedzieć, jak nazywają się neurony tego robaka
        x_flat, b_, shap_matrix, ig_matrix, _, neuron_array, _, _ = load_all_data(worm_idx=w)
        
        if target_behavior != "ALL":
            mask = (b_.flatten() == target_behavior)
            if not np.any(mask):
                # Jeśli robak nie wykonał zachowania, tworzymy puste Series
                empty_series = pd.Series(dtype=float, name=f"Worm {w}")
                worm_series_list.append(empty_series)
                continue
        else:
            mask = np.ones(len(b_.flatten()), dtype=bool)

        if target_metric == "shap":
            matrix_to_use = shap_matrix[mask]
        elif target_metric == "ig":
            matrix_to_use = ig_matrix[mask]
        else:
            matrix_to_use = x_flat[mask]

        mean_scores = np.mean(np.abs(matrix_to_use), axis=0)
        
        # Zamiast gołej listy, tworzymy obiekt Series, w którym każda wartość jest przypisana do nazwy neuronu
        worm_series = pd.Series(mean_scores, index=neuron_array, name=f"Worm {w}")
        worm_series_list.append(worm_series)

    # Łączymy wszystkie robaki w jedną tabelę. 
    # Pandas automatycznie dopasuje wiersze (neurony) o tych samych nazwach!
    df = pd.concat(worm_series_list, axis=1)
    
    # Usuwamy wiersze (neurony), których brakuje u któregokolwiek z robaków, 
    # aby móc poprawnie policzyć korelację
    df = df.dropna()

    # Liczymy korelację na dopasowanych danych
    corr_matrix = df.corr(method='spearman')

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r', 
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverinfo="x+y+z"
    ))

    metric_titles = {'shap': 'SHAP', 'ig': 'Integrated Gradients', 'activity': 'Raw Activity'}
    display_metric = metric_titles.get(target_metric, target_metric)

    fig.update_layout(
        title=f"Inter-worm Spearman Correlation<br><sup>Metric: {display_metric} | Behavior: {b_name} | Common Neurons: {len(df)}</sup>",
        xaxis=dict(title="Worms", side="bottom", tickfont=dict(size=18)),
        yaxis=dict(title="Worms", autorange="reversed", tickfont=dict(size=18)),
        width=1200,
        height=650,
        template="plotly_white",
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    return fig


def behaviour_corr(target_worm, target_metric):
    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, _, _ = load_all_data(worm_idx=target_worm)

    beh_series_list = []

    # Iterujemy przez ID (beh_idx) oraz nazwy (beh_name) ze słownika
    for beh_idx, beh_name in behavior_names.items():
        mask = (b_.flatten() == beh_idx)
        
        # Zabezpieczenie: jeśli dany robak nie wykonał tego zachowania, pomijamy je
        if not np.any(mask):
            continue

        if target_metric == "shap":
            matrix_to_use = shap_matrix[mask]
        elif target_metric == "ig":
            matrix_to_use = ig_matrix[mask]
        else:
            # Ta opcja automatycznie obsłuży "Raw Activity" (wartość "activity")
            matrix_to_use = x_flat[mask]

        mean_scores = np.mean(np.abs(matrix_to_use), axis=0)

        # Przypisujemy nazwę zachowania do kolumny (będzie widoczna na osiach wykresu)
        beh_series = pd.Series(mean_scores, index=neuron_array, name=beh_name)
        beh_series_list.append(beh_series)

    df = pd.concat(beh_series_list, axis=1)
        
    # Usuwamy wiersze (neurony), których brakuje dla któregokolwiek z wygenerowanych zachowań
    df = df.dropna()
    
    # Liczymy korelację Spearmana na dopasowanych danych
    corr_matrix = df.corr(method='spearman')
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r', 
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverinfo="x+y+z"
    ))
    
    metric_titles = {'shap': 'SHAP', 'ig': 'Integrated Gradients', 'activity': 'Raw Activity'}
    display_metric = metric_titles.get(target_metric, target_metric)
    
    fig.update_layout(
        # Zaktualizowany główny tytuł (Behavior Correlation zamiast Inter-worm)
        title=f"Inter-behavior Spearman Correlation<br><sup>Metric: {display_metric} | Worm: {target_worm} | Common Neurons: {len(df)}</sup>",
        
        # Zmienione tytuły osi na "Behaviors"
        xaxis=dict(title="Behaviors", side="bottom", tickfont=dict(size=18)),
        yaxis=dict(title="Behaviors", autorange="reversed", tickfont=dict(size=18)),
        
        width=1200,
        height=650,
        template="plotly_white",
        margin=dict(t=100, b=50, l=50, r=50)
    )
        
    return fig


def neurons_corr(target_behaviour, target_worm, target_metric, filtered_neuron, neuron_types_dict):
    
    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, _, _ = load_all_data(worm_idx=target_worm)
    
    if target_behaviour != "ALL":
        target_behaviour = int(target_behaviour)
        mask = (b_.flatten() == target_behaviour)
        b_name = behavior_names.get(target_behaviour, str(target_behaviour))
    else:
        mask = np.ones(len(b_.flatten()), dtype=bool)
        b_name = "ALL"

    
    if not np.any(mask):
        return go.Figure().update_layout(title=f"Brak danych dla zachowania: {b_name} u robaka {target_worm}")

    
    if target_metric == "shap":
        matrix_to_use = shap_matrix[mask]
    elif target_metric == "ig":
        matrix_to_use = ig_matrix[mask]
    else:
        matrix_to_use = x_flat[mask]


    df = pd.DataFrame(matrix_to_use, columns=neuron_array)
    
    corr_matrix = df.corr(method='spearman')

    y_neurons = neuron_array 

    if filtered_neuron != "ALL":
        if isinstance(filtered_neuron, list):
            # Użytkownik wybrał konkretne neurony z listy (Dropdown multi=True)
            y_neurons = [n for n in neuron_array if n in filtered_neuron]
        else:
            # Użytkownik wybrał typ neuronu (np. "Motor", "Sensory" z Twojego layoutu)
            y_neurons = [n for n in neuron_array if neuron_types_dict.get(n, 'Unidentified') == filtered_neuron]
            
        if len(y_neurons) == 0:
            return go.Figure().update_layout(title=f"Brak neuronów dla filtra: {filtered_neuron}")

    # Przycięcie macierzy do wyfiltrowanych neuronów na osi Y, ale zostawienie wszystkich na osi X
    filtered_corr_matrix = corr_matrix.loc[y_neurons, :]

    # 6. Rysowanie macierzy
    fig = go.Figure(data=go.Heatmap(
        z=filtered_corr_matrix.values,
        x=filtered_corr_matrix.columns,
        y=filtered_corr_matrix.index,
        colorscale='RdBu_r', 
        zmin=-1,
        zmax=1,
        hoverinfo="x+y+z"
        # Usunięto texttemplate, by nie zaciemniać wykresu przy tak dużej macierzy!
    ))

    metric_titles = {'shap': 'SHAP', 'ig': 'Integrated Gradients', 'activity': 'Raw Activity'}
    display_metric = metric_titles.get(target_metric, target_metric)

    fig.update_layout(
        title=f"Neuron-to-Neuron Correlation<br><sup>Metric: {display_metric} | Worm: {target_worm} | Behavior: {b_name}</sup>",
        xaxis=dict(
            title="All Neurons", 
            side="bottom",
            range=[-0.5, 39.5], # Zamraża widok na pierwszych 40 neuronach
            rangeslider=dict(visible=True, thickness=0.08) # Dodaje interaktywny suwak na dole
        ),
        yaxis=dict(
            title="Selected Neurons", 
            autorange="reversed"
        ),
        width=1400,
        # Zwiększyłam mnożnik z 15 na 25, żeby kafelki w pionie miały więcej "oddechu"
        height=max(500, len(y_neurons) * 25), 
        template="plotly_white",
        margin=dict(t=100, b=80, l=50, r=50) # Zwiększony dolny margines (b) by pomieścić suwak
    )
    
    return fig
    
    




           