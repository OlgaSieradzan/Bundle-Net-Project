# Libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Code for visualization

def temporal_dashboard(x_raw, b_raw, behavior_names, shap_matrix, ig_matrix, neuron_names, palette, title, top_n=10):

        if len(x_raw.shape) == 4:
            if x_raw.shape[-1] > x_raw.shape[0]: 
                x_flat = x_raw[:, 0, 0, :].T 
            else:
                x_flat = x_raw[:, 0, 0, :]
        elif len(x_raw.shape) == 3:
            x_flat = x_raw[:, 0, :]
        else:
            x_flat = x_raw
            
        b_flat = b_raw.flatten()
        

        min_len = min(x_flat.shape[0], len(b_flat), shap_matrix.shape[0], ig_matrix.shape[0])
        x_flat = x_flat[:min_len]
        b_flat = b_flat[:min_len]
        shap_matrix = shap_matrix[:min_len]
        ig_matrix = ig_matrix[:min_len]
        
        num_neurons = x_flat.shape[1]
        
        mean_shap = np.mean(np.abs(shap_matrix), axis=0)
        mean_ig = np.mean(np.abs(ig_matrix), axis=0)
        norm_shap = mean_shap / (np.max(mean_shap) + 1e-8)
        norm_ig = mean_ig / (np.max(mean_ig) + 1e-8)
        
        combined_mean = norm_shap + norm_ig
        sorted_idx = np.argsort(combined_mean)
        
        sorted_shap = shap_matrix[:, sorted_idx].T 
        sorted_ig = ig_matrix[:, sorted_idx].T 
        sorted_x_raw = x_flat[:, sorted_idx].T    
        sorted_names = np.array(neuron_names)[sorted_idx]

        num_classes = len(behavior_names)
        discrete_colorscale = []
        
        for i in range(num_classes):
            step_low = i / num_classes
            step_high = (i + 1) / num_classes
            c = palette[i % len(palette)]
            discrete_colorscale.append([step_low, c])
            discrete_colorscale.append([step_high, c])
            
        if isinstance(behavior_names, dict):
            b_texts = [f"Frame: {i}<br>Behavior: {behavior_names.get(int(b), str(b))}" for i, b in enumerate(b_flat)]
            ticktext = [behavior_names.get(i, str(i)) for i in range(num_classes)]
        else:
            b_texts = [f"Frame: {i}<br>Behavior: {behavior_names[int(b)]}" for i, b in enumerate(b_flat)]
            ticktext = [behavior_names[i] for i in range(num_classes)]
            
        tickvals = np.arange(num_classes) + 0.5

        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True,           
            vertical_spacing=0.03,
            row_heights=[0.35, 0.05, 0.3, 0.3], 
            subplot_titles=("Raw Neuronal Activation", "Executed Behavior", "SHAP Influence", "Integrated Gradients Influence")
        )
        
        # --- PANEL 1: Raw Activation ---
        fig.add_trace(go.Heatmap(
            z=sorted_x_raw,
            y=sorted_names,
            colorscale='magma',
            colorbar=dict(title="Activation", x=1.01, y=0.84, len=0.3, yanchor="middle") 
        ), row=1, col=1)
        
        # --- PANEL 2: Behavior ---
        fig.add_trace(go.Heatmap(
            z=[b_flat],
            colorscale=discrete_colorscale,
            zmin=0,
            zmax=num_classes,
            showscale=True, 
            colorbar=dict(
                title="Behavior",
                tickvals=tickvals,
                ticktext=ticktext,
                x=1.01,
                y=0.63,
                len=0.1, 
                yanchor="middle"
            ),
            hoverinfo="text",
            text=[b_texts]
        ), row=2, col=1)
        
        # --- PANEL 3: SHAP Heatmap ---
        max_shap = np.max(np.abs(sorted_shap))
        fig.add_trace(go.Heatmap(
            z=sorted_shap,
            y=sorted_names,
            colorscale='RdBu_r', 
            zmid=0,
            zmin=-max_shap,
            zmax=max_shap,
            colorbar=dict(title="SHAP Score", x=1.01, y=0.44, len=0.25, yanchor="middle") 
        ), row=3, col=1)
        
        # --- PANEL 4: IG Heatmap ---
        max_ig = np.max(np.abs(sorted_ig))
        fig.add_trace(go.Heatmap(
            z=sorted_ig,
            y=sorted_names,
            colorscale='RdBu_r', 
            zmid=0,
            zmin=-max_ig,
            zmax=max_ig,
            colorbar=dict(title="IG Score", x=1.01, y=0.14, len=0.25, yanchor="middle") 
        ), row=4, col=1)
        
        # 5. LAYOUT & INTERACTIVITY SETTINGS
        y_default_range = [num_neurons - top_n - 0.5, num_neurons - 0.5]
        
        fig.update_layout(
            title=title,
            height=1100,         
            width=1400,          
            autosize=False,
            margin=dict(l=50, r=150, b=50, t=50), 
            hovermode="x unified" 
        )

        fig.update_yaxes(range=y_default_range, title_text="Neurons", row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1, fixedrange=True) 
        fig.update_yaxes(range=y_default_range, title_text="Neurons", matches='y', row=3, col=1)
        fig.update_yaxes(range=y_default_range, title_text="Neurons", matches='y', row=4, col=1)
        
        fig.update_xaxes(title_text="Time Frames", row=4, col=1)

        fig.update_layout(template="plotly_white")
        
        return fig