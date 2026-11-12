
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_loader import load_all_data, METRICS_FILE, DATA_PATH, XAI_PATH
from dash import dash_table
from scipy.stats import spearmanr
from plotly.subplots import make_subplots

def inter_worm_corr(target_behavior, target_metric, num_worms=5, top_k=10, rbo_p=0.9):
    # 1. Ładowanie danych
    _, _, _, _, behavior_names, _, _, _ = load_all_data(worm_idx=0, precomputed_data_path=DATA_PATH, xai_matrices_path=XAI_PATH)
    
    if target_behavior != "ALL":
        target_behavior = int(target_behavior)
        b_name = behavior_names.get(target_behavior, str(target_behavior))
    else:
        b_name = "ALL"

    # Słownik do przechowywania posortowanych list neuronów dla każdego robaka
    worm_ranked_lists = {}
    
    for w in range(num_worms):
        x_flat, b_, shap_matrix, ig_matrix, _, neuron_array, _, _ = load_all_data(worm_idx=w, precomputed_data_path=DATA_PATH, xai_matrices_path=XAI_PATH)
        
        if target_behavior != "ALL":
            mask = (b_.flatten() == target_behavior)
            if not np.any(mask):
                continue
        else:
            mask = np.ones(len(b_.flatten()), dtype=bool)

        if target_metric == "shap":
            matrix_to_use = shap_matrix[mask]
        elif target_metric == "ig":
            matrix_to_use = ig_matrix[mask]
        else:
            matrix_to_use = x_flat[mask]

        # Uśrednianie ważności
        mean_scores = np.mean(np.abs(matrix_to_use), axis=0)
        worm_series = pd.Series(mean_scores, index=neuron_array)
        
        # Kluczowa zmiana: Zamiast trzymać surowe wartości, tworzymy posortowaną listę neuronów 
        # (od najważniejszego do najmniej ważnego)
        ranked_neurons = worm_series.sort_values(ascending=False).index.tolist()
        worm_ranked_lists[f"W{w}"] = ranked_neurons

    worms = list(worm_ranked_lists.keys())
    n = len(worms)
    
    # Macierze na wyniki
    rbo_matrix = np.zeros((n, n))
    jaccard_matrix = np.zeros((n, n))
    rbo_vals = []
    jaccard_vals = []

    # --- FUNKCJE POMOCNICZE ---
    def compute_rbo(l1, l2, p):
        if not l1 or not l2: return 0.0
        agreement = 0.0
        for d in range(1, min(len(l1), len(l2)) + 1):
            set1 = set(l1[:d])
            set2 = set(l2[:d])
            a_d = len(set1.intersection(set2)) / d
            agreement += (p ** (d - 1)) * a_d
        return (1 - p) * agreement

    def compute_jaccard(l1, l2, k):
        s1, s2 = set(l1[:k]), set(l2[:k])
        if not s1 and not s2: return 1.0
        return len(s1.intersection(s2)) / len(s1.union(s2))
    # ---------------------------

    # 2. Obliczanie metryk każdy z każdym
    for i in range(n):
        for j in range(n):
            l1 = worm_ranked_lists[worms[i]]
            l2 = worm_ranked_lists[worms[j]]
            
            rbo_score = compute_rbo(l1, l2, rbo_p)
            jac_score = compute_jaccard(l1, l2, top_k)
            
            rbo_matrix[i, j] = rbo_score
            jaccard_matrix[i, j] = jac_score
            
            # Zbieramy wartości tylko dla unikalnych par (górny trójkąt, bez przekątnej)
            if j > i:
                rbo_vals.append(rbo_score)
                jaccard_vals.append(jac_score)

    # 3. Tabela podsumowująca do Dasha
    mean_rbo = np.mean(rbo_vals) if rbo_vals else 0
    std_rbo = np.std(rbo_vals) if rbo_vals else 0
    mean_jac = np.mean(jaccard_vals) if jaccard_vals else 0
    std_jac = np.std(jaccard_vals) if jaccard_vals else 0

    table_data = [
        {"Metric": f"Mean RBO (p={rbo_p}) ± SD", "Value": f"{mean_rbo:.3f} ± {std_rbo:.3f}"},
        {"Metric": f"Mean Top-{top_k} Jaccard ± SD", "Value": f"{mean_jac:.3f} ± {std_jac:.3f}"}
    ]
    
    stats_table = dash_table.DataTable(
        data=table_data,
        columns=[{"name": i, "id": i} for i in ["Metric", "Value"]],
        style_header={'backgroundColor': '#166ea9', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'padding': '12px', 'fontSize': '16px'},
        style_as_list_view=True,
    )

    # 4. Wykres wizualny: 2 mapy ciepła obok siebie
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Rank-Biased Overlap (p={rbo_p})", f"Top-{top_k} Jaccard Similarity"))
    
    # Skala [0, 1] - od białego (brak pokrycia) do intensywnego różu (pełne pokrycie)
    colorscale_01 = [[0.0, '#ffffff'], [1.0, '#d01c8b']]
    
    fig.add_trace(go.Heatmap(
        z=rbo_matrix, x=worms, y=worms, colorscale=colorscale_01, zmin=0, zmax=1,
        text=np.round(rbo_matrix, 2), texttemplate="%{text}", textfont={"size": 16}, hoverinfo="x+y+z",
        coloraxis="coloraxis" # Łączy legendę obu wykresów
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=jaccard_matrix, x=worms, y=worms, colorscale=colorscale_01, zmin=0, zmax=1,
        text=np.round(jaccard_matrix, 2), texttemplate="%{text}", textfont={"size": 16}, hoverinfo="x+y+z",
        coloraxis="coloraxis"
    ), row=1, col=2)

    metric_titles = {'shap': 'SHAP', 'ig': 'Integrated Gradients', 'activity': 'Raw Activity'}
    display_metric = metric_titles.get(target_metric, target_metric)

    fig.update_layout(
        title=f"Inter-worm Feature Importance Consistency<br><sup>Metric: {display_metric} | Behavior: {b_name}</sup>",
        coloraxis=dict(colorscale=colorscale_01, cmin=0, cmax=1),
        width=1200, height=600, template="plotly_white", margin=dict(t=100, b=50, l=50, r=50)
    )
    
    fig.update_yaxes(autorange="reversed") # Ustawia W0 na górze, W4 na dole
    
    return fig, stats_table


def behaviour_corr(target_worm, target_metric, top_k=10, rbo_p=0.9):
    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, _, _ = load_all_data(
        worm_idx=target_worm, precomputed_data_path=DATA_PATH, xai_matrices_path=XAI_PATH
    )

    # --- Wewnętrzne funkcje do metryk ---
    def compute_rbo(l1, l2, p):
        if not l1 or not l2: return 0.0
        agreement = 0.0
        for d in range(1, min(len(l1), len(l2)) + 1):
            set1 = set(l1[:d])
            set2 = set(l2[:d])
            a_d = len(set1.intersection(set2)) / d
            agreement += (p ** (d - 1)) * a_d
        return (1 - p) * agreement

    def compute_jaccard(l1, l2, k):
        s1, s2 = set(l1[:k]), set(l2[:k])
        if not s1 and not s2: return 1.0
        return len(s1.intersection(s2)) / len(s1.union(s2))
    # ------------------------------------

    beh_ranked_lists = {}
    valid_b_names = []

    # 1. Wyciąganie i sortowanie neuronów dla każdego zachowania
    for beh_idx, beh_name in behavior_names.items():
        mask = (b_.flatten() == beh_idx)

        if not np.any(mask):
            continue
            
        valid_b_names.append(beh_name)

        if target_metric == "shap":
            matrix_to_use = shap_matrix[mask]
        elif target_metric == "ig":
            matrix_to_use = ig_matrix[mask]
        else:
            matrix_to_use = x_flat[mask]

        mean_scores = np.mean(np.abs(matrix_to_use), axis=0)
        beh_series = pd.Series(mean_scores, index=neuron_array)
        
        # Sortowanie od najważniejszego do najmniej ważnego
        ranked_neurons = beh_series.sort_values(ascending=False).index.tolist()
        beh_ranked_lists[beh_name] = ranked_neurons

    n_beh = len(valid_b_names)
    rbo_matrix = np.zeros((n_beh, n_beh))
    jaccard_matrix = np.zeros((n_beh, n_beh))

    # 2. Obliczanie metryk RBO i Jaccard każdy-z-każdym
    for i in range(n_beh):
        for j in range(n_beh):
            l1 = beh_ranked_lists[valid_b_names[i]]
            l2 = beh_ranked_lists[valid_b_names[j]]
            
            rbo_matrix[i, j] = compute_rbo(l1, l2, rbo_p)
            jaccard_matrix[i, j] = compute_jaccard(l1, l2, top_k)

    # 3. Rysowanie dwupanelowego wykresu (Subplots)
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f"Rank-Biased Overlap (p={rbo_p})", f"Top-{top_k} Jaccard Similarity"),
        horizontal_spacing=0.1
    )
    
    # Skala 0-1, od białego do różu
    colorscale_01 = [[0.0, '#ffffff'], [1.0, '#d01c8b']]

    fig.add_trace(go.Heatmap(
        z=rbo_matrix, x=valid_b_names, y=valid_b_names, colorscale=colorscale_01, zmin=0, zmax=1,
        text=np.round(rbo_matrix, 2), texttemplate="%{text}", textfont={"size": 16}, hoverinfo="x+y+z",
        coloraxis="coloraxis"
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=jaccard_matrix, x=valid_b_names, y=valid_b_names, colorscale=colorscale_01, zmin=0, zmax=1,
        text=np.round(jaccard_matrix, 2), texttemplate="%{text}", textfont={"size": 16}, hoverinfo="x+y+z",
        coloraxis="coloraxis"
    ), row=1, col=2)

    metric_titles = {'shap': 'SHAP', 'ig': 'Integrated Gradients', 'activity': 'Raw Activity'}
    display_metric = metric_titles.get(target_metric, target_metric)
    
    # 4. Aktualizacja layoutu
    fig.update_layout(
        title=dict(
            text=f"Inter-behavior Consistency<br><sup>Metric: {display_metric} | Worm: {target_worm}</sup>",
            font=dict(size=22)
        ),
        coloraxis=dict(colorscale=colorscale_01, cmin=0, cmax=1),
        width=1400, 
        height=700, 
        template="plotly_white", 
        margin=dict(t=100, b=100, l=50, r=50)
    )
    
    # Odwrócenie osi Y oraz powiększenie czcionek osi
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=14))
    fig.update_xaxes(tickfont=dict(size=14))
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=18)

    # 5. Generowanie tabeli (wykorzystanie Twojej istniejącej funkcji)
    matrix_df = compute_behavior_vs_behavior_matrix(target_metric=target_metric, num_worms=5)
    matrix_df = matrix_df.reset_index().rename(columns={'index': 'Behavior'})

    dash_matrix_table = dash_table.DataTable(
        data=matrix_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in matrix_df.columns],
        style_header={
            'backgroundColor': '#166ea9',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_cell={'textAlign': 'center', 'padding': '12px'},
        style_data_conditional=[
            {
                'if': {'column_id': 'Behavior'},
                'fontWeight': 'bold',
                'backgroundColor': '#f8f9fa'
            }
        ],
        style_as_list_view=True,
    )

    return fig, dash_matrix_table
        



def neurons_corr(target_behaviour, target_worm, target_metric, filtered_neuron_type, filtered_neuron_names, neuron_types_dict):
    
    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, _, _ = load_all_data(worm_idx=target_worm, precomputed_data_path= DATA_PATH, xai_matrices_path= XAI_PATH)
    
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

    
    if filtered_neuron_type != "ALL":
        y_neurons = [n for n in y_neurons if neuron_types_dict.get(n, 'Unidentified') == filtered_neuron_type]
        
    
    if isinstance(filtered_neuron_names, list) and len(filtered_neuron_names) > 0:
        
        y_neurons = [n for n in y_neurons if n in filtered_neuron_names]

    if len(y_neurons) == 0:
        return go.Figure().update_layout(title="Brak wybranych neuronów do wyświetlenia.")

    filtered_corr_matrix = corr_matrix.loc[y_neurons, :]

    # 6. Rysowanie macierzy
    fig = go.Figure(data=go.Heatmap(
        z=filtered_corr_matrix.values,
        x=filtered_corr_matrix.columns,
        y=filtered_corr_matrix.index,
        colorscale=[
            [0.0, '#1f77b4'],  
            [0.5, '#ffffff'], 
            [1.0, '#d01c8b']   
        ],
        zmin=-1,
        zmax=1,
        text=np.round(filtered_corr_matrix.values, 2),
        texttemplate="%{text}",
        hoverinfo="x+y+z"
    ))

    metric_titles = {'shap': 'SHAP', 'ig': 'Integrated Gradients', 'activity': 'Raw Activity'}
    display_metric = metric_titles.get(target_metric, target_metric)

    fig.update_layout(
        title=f"Neuron-to-Neuron Correlation<br><sup>Metric: {display_metric} | Worm: {target_worm} | Behavior: {b_name}</sup>",
        xaxis=dict(
            title="All Neurons", 
            side="bottom",
            range=[-0.5, 39.5], 
            rangeslider=dict(visible=True, thickness=0.02) 
        ),
        yaxis=dict(
            title="Selected Neurons", 
            autorange="reversed"
        ),
        width=1400,
        
        height=max(500, len(y_neurons) * 25), 
        template="plotly_white",
        margin=dict(t=100, b=80, l=50, r=50) 
    )
    
    return fig
    
    
def generate_global_boxplot_summary(num_worms=5, top_k=10, rbo_p=0.9):
    
    # Wewnętrzne funkcje do obliczania metryk
    def compute_rbo(l1, l2, p):
        if not l1 or not l2: return 0.0
        agreement = 0.0
        for d in range(1, min(len(l1), len(l2)) + 1):
            set1 = set(l1[:d])
            set2 = set(l2[:d])
            a_d = len(set1.intersection(set2)) / d
            agreement += (p ** (d - 1)) * a_d
        return (1 - p) * agreement

    def compute_jaccard(l1, l2, k):
        s1, s2 = set(l1[:k]), set(l2[:k])
        if not s1 and not s2: return 1.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    behavior_map = {
        'ALL': 'Global (All)',
        0: 'Forward',
        1: 'Reverse-1',
        2: 'Reverse-2',
        3: 'Sustained Reversal',
        4: 'Slowing',
        5: 'Dorsal Turn',
        6: 'Ventral Turn'
    }

    # Ładowanie danych dla wszystkich robaków
    worms_data = {}
    for w in range(num_worms):
        x_flat, b_, shap_matrix, ig_matrix, _, neuron_array, _, _ = load_all_data(
            worm_idx=w, precomputed_data_path=DATA_PATH, xai_matrices_path=XAI_PATH
        )
        worms_data[w] = {
            'b': b_.flatten(),
            'shap': shap_matrix,
            'ig': ig_matrix,
            'neurons': neuron_array
        }

    fig = make_subplots(
        rows=2, cols=4, 
        subplot_titles=list(behavior_map.values()),
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )

    metrics = ['shap', 'ig']
    
    # Konfiguracja wizualna 4 pudełek na każdym wykresie (RBO i Jaccard dla SHAP i IG)
    trace_configs = {
        'shap_rbo': {'name': 'SHAP (RBO)', 'color': '#8B0A50'},               # Ciemny róż
        'shap_jac': {'name': f'SHAP (Top-{top_k} Jaccard)', 'color': '#d01c8b'}, # Jasny róż
        'ig_rbo': {'name': 'IG (RBO)', 'color': '#00688B'},                   # Ciemny niebieski
        'ig_jac': {'name': f'IG (Top-{top_k} Jaccard)', 'color': '#7AC5CD'}      # Jasny niebieski
    }

    row, col = 1, 1
    for b_key, b_name in behavior_map.items():
        
        # Słownik na wyniki pairwise dla obecnego zachowania
        results = {'shap_rbo': [], 'shap_jac': [], 'ig_rbo': [], 'ig_jac': []}
        
        for metric in metrics:
            worm_ranked_lists = []
            
            for w in range(num_worms):
                w_data = worms_data[w]
                
                if b_key != 'ALL':
                    mask = (w_data['b'] == b_key)
                else:
                    mask = np.ones(len(w_data['b']), dtype=bool)
                
                if not np.any(mask):
                    worm_ranked_lists.append([])
                    continue
                    
                matrix_to_use = w_data[metric][mask]
                mean_scores = np.mean(np.abs(matrix_to_use), axis=0)
                worm_series = pd.Series(mean_scores, index=w_data['neurons'])
                
                # Zamiast surowych wartości (jak w Spearmanie), zapisujemy posortowaną hierarchię
                ranked_neurons = worm_series.sort_values(ascending=False).index.tolist()
                worm_ranked_lists.append(ranked_neurons)

            # Indeksy robaków, które faktycznie wykonały dany ruch
            valid_worms = [idx for idx, lst in enumerate(worm_ranked_lists) if len(lst) > 0]

            # Obliczanie kombinacji każdy-z-każdym
            for i in range(len(valid_worms)):
                for j in range(i + 1, len(valid_worms)):
                    l1 = worm_ranked_lists[valid_worms[i]]
                    l2 = worm_ranked_lists[valid_worms[j]]
                    
                    results[f"{metric}_rbo"].append(compute_rbo(l1, l2, rbo_p))
                    results[f"{metric}_jac"].append(compute_jaccard(l1, l2, top_k))

        # Dodawanie 4 śladów (traces) do obecnego pod-wykresu
        for config_key, config in trace_configs.items():
            vals = results[config_key]
            if vals:
                fig.add_trace(
                    go.Box(
                        y=vals,
                        x=[config['name']] * len(vals),
                        name=config['name'],
                        marker_color=config['color'],
                        boxpoints='all', 
                        jitter=0.1,
                        width=0.6,
                        showlegend=(row == 1 and col == 1) 
                    ),
                    row=row, col=col
                )

        col += 1
        if col > 4:
            col = 1
            row += 1

    # Obie metryki RBO i Jaccard zawierają się sztywno w przedziale [0, 1]
    fig.update_yaxes(
        range=[-0.05, 1.05], 
        title_text="Score" if col==1 else "",
        title_font=dict(size=18),  # Większy podpis osi Y
        tickfont=dict(size=16)     # Większe cyferki na osi Y
    )
    fig.update_xaxes(showticklabels=False) 
    
    # 2. Zwiększenie czcionki dla tytułów poszczególnych paneli (Forward, Reverse-1 itd.)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=18)

    # 3. Zwiększenie czcionki tytułu głównego, legendy i globalnych ustawień
    fig.update_layout(
        font=dict(size=16),        # Globalny rozmiar (np. dla dymków po najechaniu myszką)
        height=850,                # Lekko zwiększona wysokość, by pomieścić większe napisy
        width=1400,
        boxmode='group', 
        template="plotly_white",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.06,                # Przesunięto legendę wyżej, by nie nachodziła na tytuły
            xanchor="right", 
            x=1,
            font=dict(size=18)     # Większa czcionka w legendzie
        )
    )

    return fig


def compute_behavior_vs_behavior_matrix(target_metric="shap", num_worms=5, top_k=10, rbo_p=0.9):
    
    # --- Wewnętrzne funkcje do metryk ---
    def compute_rbo(l1, l2, p):
        if not l1 or not l2: return 0.0
        agreement = 0.0
        for d in range(1, min(len(l1), len(l2)) + 1):
            set1 = set(l1[:d])
            set2 = set(l2[:d])
            a_d = len(set1.intersection(set2)) / d
            agreement += (p ** (d - 1)) * a_d
        return (1 - p) * agreement

    def compute_jaccard(l1, l2, k):
        s1, s2 = set(l1[:k]), set(l2[:k])
        if not s1 and not s2: return 1.0
        return len(s1.intersection(s2)) / len(s1.union(s2))
    # ------------------------------------

    behaviors = [0, 1, 2, 3, 4, 5, 6]
    b_names = ['Fwd', 'Rev-1', 'Rev-2', 'Sust-Rev', 'Slow', 'Dorsal', 'Ventral']

    # Zmieniony słownik: przechowujemy listy wyników RBO i Jaccard dla każdej pary
    pair_stats = { (i, j): {'rbo': [], 'jac': []} for i in behaviors for j in behaviors }
    
    for w in range(num_worms):
        x_flat, b_, shap_matrix, ig_matrix, _, neuron_array, _, _ = load_all_data(w, DATA_PATH, XAI_PATH)
        b_flat = b_.flatten()
        matrix = shap_matrix if target_metric == "shap" else ig_matrix

        b_ranked_lists = {}
        for b_idx in behaviors:
            mask = (b_flat == b_idx)
            if np.any(mask):
                mean_scores = np.mean(np.abs(matrix[mask]), axis=0)
                beh_series = pd.Series(mean_scores, index=neuron_array)
                # Zapisujemy wyselekcjonowane i posortowane rankingi neuronów
                b_ranked_lists[b_idx] = beh_series.sort_values(ascending=False).index.tolist()
            else:
                b_ranked_lists[b_idx] = None # Robak nie wykonał tego ruchu
                
        # Obliczanie metryk każdy z każdym dla danego robaka
        for i in behaviors:
            for j in behaviors:
                if b_ranked_lists[i] is not None and b_ranked_lists[j] is not None:
                    l1 = b_ranked_lists[i]
                    l2 = b_ranked_lists[j]
                    
                    rbo_val = compute_rbo(l1, l2, rbo_p)
                    jac_val = compute_jaccard(l1, l2, top_k)
                    
                    pair_stats[(i, j)]['rbo'].append(rbo_val)
                    pair_stats[(i, j)]['jac'].append(jac_val)
    
    # Budowanie ostatecznej macierzy
    matrix_df = pd.DataFrame(index=b_names, columns=b_names)
    
    for idx_i, i in enumerate(behaviors):
        for idx_j, j in enumerate(behaviors):
            rbo_vals = pair_stats[(i, j)]['rbo']
            jac_vals = pair_stats[(i, j)]['jac']
            
            if len(rbo_vals) > 0:
                mean_rbo = np.mean(rbo_vals)
                mean_jac = np.mean(jac_vals)
                
                if i == j:
                    matrix_df.iat[idx_i, idx_j] = "1.00" # Na przekątnej idealne pokrycie
                else:
                    # Format wyświetlający obie metryki (np. "0.45 | J: 0.30")
                    matrix_df.iat[idx_i, idx_j] = f"{mean_rbo:.2f} | J: {mean_jac:.2f}"
            else:
                matrix_df.iat[idx_i, idx_j] = "N/A"
                
    return matrix_df


def generate_shap_vs_ig_scatter(num_worms=5, rbo_p=0.9):
    
    def compute_rbo(l1, l2, p):
        if not l1 or not l2: return 0.0
        agreement = 0.0
        for d in range(1, min(len(l1), len(l2)) + 1):
            set1 = set(l1[:d])
            set2 = set(l2[:d])
            a_d = len(set1.intersection(set2)) / d
            agreement += (p ** (d - 1)) * a_d
        return (1 - p) * agreement

    behavior_map = {
        'ALL': 'Global (All)', 0: 'Forward', 1: 'Reverse-1', 
        2: 'Reverse-2', 3: 'Sustained Reversal', 4: 'Slowing', 
        5: 'Dorsal Turn', 6: 'Ventral Turn'
    }

    # Wczytanie danych
    worms_data = {}
    for w in range(num_worms):
        x_flat, b_, shap_matrix, ig_matrix, _, neuron_array, _, _ = load_all_data(
            worm_idx=w, precomputed_data_path=DATA_PATH, xai_matrices_path=XAI_PATH
        )
        worms_data[w] = {
            'b': b_.flatten(), 'shap': shap_matrix, 'ig': ig_matrix, 'neurons': neuron_array
        }

    fig = make_subplots(
        rows=2, cols=4, 
        subplot_titles=list(behavior_map.values()),
        vertical_spacing=0.15, horizontal_spacing=0.08
    )

    row, col = 1, 1
    for b_key, b_name in behavior_map.items():
        shap_rbo_vals = []
        ig_rbo_vals = []
        
        worm_ranked_shap = []
        worm_ranked_ig = []
        
        # Wyciąganie rankingów dla SHAP i IG
        for w in range(num_worms):
            w_data = worms_data[w]
            mask = (w_data['b'] == b_key) if b_key != 'ALL' else np.ones(len(w_data['b']), dtype=bool)
            
            if not np.any(mask):
                worm_ranked_shap.append([])
                worm_ranked_ig.append([])
                continue
                
            for metric, lst in [('shap', worm_ranked_shap), ('ig', worm_ranked_ig)]:
                matrix_to_use = w_data[metric][mask]
                mean_scores = np.mean(np.abs(matrix_to_use), axis=0)
                series = pd.Series(mean_scores, index=w_data['neurons'])
                lst.append(series.sort_values(ascending=False).index.tolist())

        valid_worms = [idx for idx, lst in enumerate(worm_ranked_shap) if len(lst) > 0]

        # Obliczanie RBO dla par
        for i in range(len(valid_worms)):
            for j in range(i + 1, len(valid_worms)):
                w_i, w_j = valid_worms[i], valid_worms[j]
                
                shap_val = compute_rbo(worm_ranked_shap[w_i], worm_ranked_shap[w_j], rbo_p)
                ig_val = compute_rbo(worm_ranked_ig[w_i], worm_ranked_ig[w_j], rbo_p)
                
                shap_rbo_vals.append(shap_val)
                ig_rbo_vals.append(ig_val)

        # Rysowanie punktów
        if shap_rbo_vals:
            fig.add_trace(
                go.Scatter(
                    x=ig_rbo_vals, y=shap_rbo_vals,
                    mode='markers',
                    marker=dict(size=10, color='#8B0A50', opacity=0.8, line=dict(width=1, color='white')),
                    name=b_name,
                    showlegend=False,
                    hovertemplate="IG: %{x:.2f}<br>SHAP: %{y:.2f}<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Linia y = x (punkty powyżej tej linii oznaczają przewagę SHAP)
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray', width=1.5),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

        # Ustawienia osi dla każdego sub-wykresu
        fig.update_xaxes(range=[0, 1], title_text="IG (RBO)" if row==2 else "", row=row, col=col)
        fig.update_yaxes(range=[0, 1], title_text="SHAP (RBO)" if col==1 else "", row=row, col=col)

        col += 1
        if col > 4:
            col = 1
            row += 1

    fig.update_yaxes(
            range=[-0.05, 1.05], 
            title_text="Score" if col==1 else "",
            title_font=dict(size=18),  # Większy podpis osi Y
            tickfont=dict(size=16)     # Większe cyferki na osi Y
        )
    fig.update_xaxes(showticklabels=False) 
    
    # 2. Zwiększenie czcionki dla tytułów poszczególnych paneli (Forward, Reverse-1 itd.)
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=18)

    # 3. Zwiększenie czcionki tytułu głównego, legendy i globalnych ustawień
    fig.update_layout(
        font=dict(size=16),        # Globalny rozmiar (np. dla dymków po najechaniu myszką)
        height=850,                # Lekko zwiększona wysokość, by pomieścić większe napisy
        width=1400,
        boxmode='group', 
        template="plotly_white",
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.06,                # Przesunięto legendę wyżej, by nie nachodziła na tytuły
            xanchor="right", 
            x=1,
            font=dict(size=18)     # Większa czcionka w legendzie
        )
    )

    return fig        


def generate_inter_behavior_variance_matrix(num_worms=5, rbo_p=0.9):
    
    def compute_rbo(l1, l2, p):
        if not l1 or not l2: return 0.0
        agreement = 0.0
        for d in range(1, min(len(l1), len(l2)) + 1):
            set1 = set(l1[:d])
            set2 = set(l2[:d])
            a_d = len(set1.intersection(set2)) / d
            agreement += (p ** (d - 1)) * a_d
        return (1 - p) * agreement

    behaviors = [0, 1, 2, 3, 4, 5, 6]
    b_names = ['Fwd', 'Rev-1', 'Rev-2', 'Sust-Rev', 'Slow', 'Dorsal', 'Ventral']
    
    # Słownik przechowujący wyniki dla każdej pary zachowań
    # Klucz: (i, j), Wartość: {'shap': [5 wartości], 'ig': [5 wartości]}
    pair_data = { (i, j): {'shap': [], 'ig': []} for i in behaviors for j in behaviors if i > j }
    
    # 1. Obliczanie RBO dla każdego robaka i każdej pary zachowań
    for w in range(num_worms):
        x_flat, b_, shap_matrix, ig_matrix, _, neuron_array, _, _ = load_all_data(w, DATA_PATH, XAI_PATH)
        b_flat = b_.flatten()
        
        b_ranked_shap = {}
        b_ranked_ig = {}
        
        for b_idx in behaviors:
            mask = (b_flat == b_idx)
            if np.any(mask):
                # SHAP
                mean_shap = np.mean(np.abs(shap_matrix[mask]), axis=0)
                b_ranked_shap[b_idx] = pd.Series(mean_shap, index=neuron_array).sort_values(ascending=False).index.tolist()
                
                # IG
                mean_ig = np.mean(np.abs(ig_matrix[mask]), axis=0)
                b_ranked_ig[b_idx] = pd.Series(mean_ig, index=neuron_array).sort_values(ascending=False).index.tolist()
            else:
                b_ranked_shap[b_idx] = None
                b_ranked_ig[b_idx] = None

        # Obliczanie kombinacji dolnego trójkąta (i > j)
        for i in behaviors:
            for j in behaviors:
                if i > j: 
                    if b_ranked_shap[i] and b_ranked_shap[j]:
                        pair_data[(i, j)]['shap'].append(compute_rbo(b_ranked_shap[i], b_ranked_shap[j], rbo_p))
                    if b_ranked_ig[i] and b_ranked_ig[j]:
                        pair_data[(i, j)]['ig'].append(compute_rbo(b_ranked_ig[i], b_ranked_ig[j], rbo_p))

    # 2. Tworzenie macierzy wykresów (7 wierszy, 7 kolumn)
    fig = make_subplots(
        rows=7, cols=7, 
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.02, horizontal_spacing=0.02
    )

    # 3. Wypełnianie komórek macierzy punktami robaków
    for i in range(7):
        for j in range(7):
            row = i + 1
            col = j + 1
            
            if i > j: # Rysujemy tylko dolny trójkąt
                shap_vals = pair_data[(i, j)]['shap']
                ig_vals = pair_data[(i, j)]['ig']
                
                # Punkty SHAP
                if shap_vals:
                    fig.add_trace(go.Box(
                        y=shap_vals, name='SHAP',
                        boxpoints='all', jitter=0.3, pointpos=0,
                        marker_color='#8B0A50', line_color='rgba(208, 28, 139, 0.4)',
                        fillcolor='rgba(208, 28, 139, 0.1)', width=0.5,
                        showlegend=(i==1 and j==0) # Pokazujemy w legendzie tylko raz
                    ), row=row, col=col)
                    
                # Punkty IG
                if ig_vals:
                    fig.add_trace(go.Box(
                        y=ig_vals, name='IG',
                        boxpoints='all', jitter=0.3, pointpos=0,
                        marker_color='#00688B', line_color='rgba(31, 119, 180, 0.4)',
                        fillcolor='rgba(31, 119, 180, 0.1)', width=0.5,
                        showlegend=(i==1 and j==0)
                    ), row=row, col=col)
                    
            # Ukrywanie linii siatki i osi w pustych komórkach nad przekątną
            elif i <= j:
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)

            # Ustawianie stałego zakresu [0,1] dla RBO
            if i > j:
                fig.update_yaxes(range=[-0.05, 1.05], showticklabels=(col==1), row=row, col=col)
                fig.update_xaxes(showticklabels=False, row=row, col=col)
                
            # Podpisy rzędów (po lewej stronie) i kolumn (na dole)
            if col == 1 and i > 0:
                fig.update_yaxes(title_text=b_names[i], title_font=dict(size=20, color='black'), row=row, col=col)
            if row == 7 and j < 6:
                fig.update_xaxes(title_text=b_names[j], title_font=dict(size=20, color='black'), row=row, col=col)

    fig.update_layout(
        height=1200,
        width=1400,
        boxmode='group',
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=22)),
        margin=dict(t=100, b=80, l=80, r=40)
    )

    return fig