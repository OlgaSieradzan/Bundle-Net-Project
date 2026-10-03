# Libraries

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from data_loader import load_all_data

from panel_global_importance import global_importance_barplot, comparison_table_xai, shap_beeswarm
from panel_heatmap import temporal_dashboard
from panel_worm_specific import latent_space_3D, empty_xai_bar_plots, CLICK_xai_barplots

x_flat, b_, shap_matrix, ig_matrix, latent_Y, behavior_names, neuron_array, palette = load_all_data(worm_idx=0)

# 2. Inicjalizacja Aplikacji
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("BunDLe-Net XAI Dashboard", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    
    dcc.Tabs(id="tabs-main", value='tab-global', children=[
        dcc.Tab(label='Global Importance', value='tab-global'),
        dcc.Tab(label='Worm Specific (3D XAI)', value='tab-specific'),
        dcc.Tab(label='Temporal Dashboard', value='tab-temporal'),
    ]),
    
    html.Div(id='tabs-content', style={'padding': '20px'})
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-main', 'value')
)
def render_content(tab):
    if tab == 'tab-global':
        fig = global_importance_barplot(shap_matrix, ig_matrix, neuron_array)
        return html.Div([dcc.Graph(figure=fig)])
        
    elif tab == 'tab-temporal':
        fig = temporal_dashboard(x_flat, b_, behavior_names, shap_matrix, ig_matrix, neuron_array, palette)
        return html.Div([dcc.Graph(figure=fig)])
        
    elif tab == 'tab-specific':
        fig_3d = latent_space_3D(latent_Y, b_, behavior_names, palette)
        fig_xai_empty = empty_xai_bar_plots()
        
        return html.Div([
            html.Div(dcc.Graph(id='plot-3d', figure=fig_3d), style={'width': '48%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(id='plot-xai', figure=fig_xai_empty), style={'width': '48%', 'display': 'inline-block'})
        ])


@app.callback(
    Output('plot-xai', 'figure'),
    Input('plot-3d', 'clickData'),
    prevent_initial_call=True
)
def update_xai_on_click(clickData):
    if clickData is None:
        return dash.no_update

    t_idx = clickData['points'][0]['customdata']
    
    # Wywołujemy funkcję z pliku visualizations.py
    return CLICK_xai_barplots(t_idx, shap_matrix, ig_matrix, neuron_array, b_, behavior_names)

if __name__ == '__main__':
    app.run_server(debug=True)