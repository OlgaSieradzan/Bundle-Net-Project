# Libraries

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from data_loader import load_all_data

from panel_global_importance import global_importance_barplot, comparison_table_xai, shap_beeswarm
from panel_heatmap import temporal_dashboard
from panel_worm_specific import latent_space_3D, empty_xai_bar_plots, CLICK_xai_barplots

x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=0)

c_elegans_categories = {
    "Sensory": [
        "ALML", "ALMR", "AVM", "PLML", "PLMR", "PVM", "ASHL", "ASHR", "AWAL", "AWAR", 
        "AWBL", "AWBR", "AWCL", "AWCR", "ASEL", "ASER", "AFDL", "AFDR", "ASJL", "ASJR",
        "ASKL", "ASKR", "ADLL", "ADLR", "ADEL", "ADER", "PDEL", "PDER", "CEPVL", "CEPVR"
    ],
    "Motor": [
        "SMDDL", "SMDDR", "SMDVL", "SMDVR", "RMDDL", "RMDDR", "RMDVL", "RMDVR", "RMED", 
        "RMEV", "DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7", "DA8", "DA9",
        "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",
        "VA1", "VA2", "VA3", "VA4", "VA5", "VA6", "VA7", "VA8", "VA9", "VA10", "VA11", "VA12",
        "VB1", "VB2", "VB3", "VB4", "VB5", "VB6", "VB7", "VB8", "VB9", "VB10", "VB11",
        "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "VD1", "VD2", "VD3", "VD4", "VD5", "VD6"
    ],
    "Interneuron": [
        "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR", 
        "DVA", "DVC", "AIBL", "AIBR", "AIYL", "AIYR", "AIZL", "AIZR", "RIAL", "RIAR", 
        "RIBL", "RIBR", "RICL", "RICR", "RIML", "RIMR", "RMGL", "RMGR", "PVNL", "PVNR"
    ]
}

neuron_types_dict = {}
for n_type, neurons in c_elegans_categories.items():
    for neuron in neurons:
        neuron_types_dict[neuron] = n_type


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
    behavior_options = [{'label': 'Global (Whole time record)', 'value': 'ALL'}] + \
                   [{'label': name, 'value': idx} for idx, name in behavior_names.items()]

    if tab == 'tab-global':
        return html.Div([
            # FILTERING PANEL
            html.Div([
                html.Div([
                    html.Label("Worm:"),
                    dcc.Dropdown(
                        id='dropdown-worm',
                        options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                        value=0, 
                        clearable=False
                    )
                ], style={'width': '45%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Behaviour:"),
                    dcc.Dropdown(
                        id='dropdown-behavior',
                        options=behavior_options,
                        value='ALL', 
                        clearable=False
                    )
                ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'}),

                html.Div([
                html.Label("Neuron type:"),
                dcc.Dropdown(
                    id='dropdown-neuron-type',
                    options=[
                        {'label': 'All', 'value': 'ALL'},
                        {'label': 'Sensory', 'value': 'Sensory'},
                        {'label': 'Motor', 'value': 'Motor'},
                        {'label': 'Interneuron', 'value': 'Interneuron'},
                        {'label': 'Unidentified', 'value': 'Unidentified'}
                        
                    ],
                    value='ALL', 
                    clearable=False
                )
            ], style={'width': '32%', 'display': 'inline-block', 'marginLeft': '2%'})

            ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px', 'borderRadius': '5px'}),
            
            # PLOTS
            html.Div([
                #  LEFT COLUMN
                html.Div([
                    html.Img(id='global-beeswarm', style={'maxWidth': '100%', 'display': 'block', 'margin': 'auto'})
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # RIGHT COLUMN
                html.Div([
                    dcc.Graph(id='global-bar-plot'),
                    html.Div(dcc.Graph(id='global-table'), style={'marginTop': '20px'})
                ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
                
            ], style={'width': '100%'})
        ])
        
    elif tab == 'tab-temporal':
        fig = temporal_dashboard(x_flat, b_, behavior_names, shap_matrix, ig_matrix, neuron_array, palette, title= "Thermal dashboard")
        return html.Div([dcc.Graph(figure=fig)])
        
    elif tab == 'tab-specific':
        fig_3d = latent_space_3D(x_flat, model ,b_,  behavior_names, palette)
        fig_xai_empty = empty_xai_bar_plots()

        
        return html.Div([
            html.Div(dcc.Graph(id='plot-3d', figure=fig_3d), style={'width': '48%', 'display': 'inline-block'}),
            html.Div(dcc.Graph(id='plot-xai', figure=fig_xai_empty), style={'width': '48%', 'display': 'inline-block'})
        ])


@app.callback(
    [Output('global-bar-plot', 'figure'),
     Output('global-table', 'figure'),
     Output('global-beeswarm', 'src')],
    [Input('dropdown-worm', 'value'),
     Input('dropdown-behavior', 'value'),
     Input('dropdown-neuron-type', 'value')
     ]
)
def update_global_dashboard(selected_worm, selected_behavior, selected_neuron_type):

    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=selected_worm)
    target_b = None if selected_behavior == 'ALL' else selected_behavior

    if selected_neuron_type != 'ALL':
        valid_indices = []
        
        for i, name in enumerate(neuron_array):
            current_type = neuron_types_dict.get(name, 'Unidentified')
            if current_type == selected_neuron_type:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            import plotly.graph_objects as go
            empty_fig = go.Figure().update_layout(title=f"Brak neuronów typu: {selected_neuron_type}")
            return empty_fig, empty_fig, ""

        neuron_array = neuron_array[valid_indices]
        shap_matrix = shap_matrix[:, valid_indices]
        ig_matrix = ig_matrix[:, valid_indices]
        x_flat = x_flat[:, valid_indices]


    fig_bars = global_importance_barplot(shap_matrix, ig_matrix, b_, behavior_names, neuron_array, target_b) 
    fig_table = comparison_table_xai(shap_matrix, ig_matrix, b_, behavior_names, neuron_array, target_behavior=target_b)
    beeswarm_src = shap_beeswarm(shap_matrix, x_flat, b_, behavior_names, neuron_array, target_behavior=target_b)
    
    return fig_bars, fig_table, beeswarm_src

@app.callback(
    Output('plot-xai', 'figure'),
    Input('plot-3d', 'clickData'),
    prevent_initial_call=True
)

def update_xai_on_click(clickData):
    if clickData is None:
        return dash.no_update

    point_data = clickData['points'][0]

    if point_data.get('curveNumber') != 0:
        return dash.no_update

    t_idx = point_data.get('pointNumber', point_data.get('customdata'))

    if isinstance(t_idx, (list, tuple)):
        t_idx = t_idx[0]

    return CLICK_xai_barplots(t_idx, shap_matrix, ig_matrix, neuron_array, b_, behavior_names)

if __name__ == '__main__':
    app.run(debug=True)