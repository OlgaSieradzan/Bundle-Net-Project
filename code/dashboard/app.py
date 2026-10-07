# Paths
import os
import sys
import numpy as np
import json
from pathlib import Path
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Dodajemy go do środowiska Pythona
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Libraries

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from data_loader import load_all_data

from panel_global_importance import global_importance_barplot, comparison_table_xai, shap_beeswarm
from panel_heatmap import temporal_dashboard
from panel_worm_specific import latent_space_3D, empty_xai_bar_plots, CLICK_xai_barplots
from xai_methods import BehaviorWrapper


# Dictionary for neurons
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

#Quality metrics 
METRICS_FILE = "C:/Users/olgas/GitHub/Bundle-Net-Project/code/precomputed_data/quality_metrics.json"

with open(METRICS_FILE, 'r') as f:
    quality_metrics = json.load(f)



# One worms data 
x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=0)


################## LAYOUT #############################################
app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id="url"),

    html.Div([
        html.Button(
            "☰", 
            id="btn-open-sidebar", 
            style={'fontSize': '30px', 'background': 'transparent', 'border': 'none', 'cursor': 'pointer', 'color': '#2C3E50', 'marginRight': '20px'}
        ),
        html.H2(id="page-title", style={'margin': '0', 'color': '#2C3E50', 'fontWeight': 'bold'})
    ], style={
        'display': 'flex', 'alignItems': 'center', 'padding': '15px 30px', 
        'backgroundColor': '#ffffff', 'borderBottom': '2px solid #ecf0f1', 
        'boxShadow': '0px 2px 5px rgba(0,0,0,0.05)'
    }),

    dbc.Offcanvas(
        html.Div([
            html.P("Choose your panel:", style={'color': '#7f8c8d'}),
            dbc.Nav([
                dbc.NavLink("Global Panel", href="/global", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'}),
                dbc.NavLink("Worm Specific Panel", href="/worm", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'}),
                dbc.NavLink("Temporal Plot Panel", href="/temp", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'}),
                dbc.NavLink("Quality check summary", href="/quality", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'})
            ], vertical=True, pills=True),
        ]),
        id="sidebar-offcanvas",
        title="Menu",
        is_open=False, 
        style={'width': '300px'}
    ),

    html.Div(id="page-content", style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'minHeight': '100vh'})
])

###################### CALLBACKS ####################################################

########## OPERATING BETWEEN TABS ###################################################
@app.callback(
    Output("sidebar-offcanvas", "is_open"),
    Input("btn-open-sidebar", "n_clicks"),
    [State("sidebar-offcanvas", "is_open")],
)
def toggle_sidebar(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

########## RENDERING CONTENT FOR EACH TAB ###########################################

@app.callback(
    [Output("page-content", "children"),
     Output("page-title", "children")],
    [Input("url", "pathname")]
)
def render_content(pathname):
    behavior_options = [{'label': 'Global (Whole time record)', 'value': 'ALL'}] + \
                   [{'label': name, 'value': idx} for idx, name in behavior_names.items()]

    if pathname == "/" or pathname == "/global":
        return html.Div([
            # FILTERING PANEL
            html.Div([
                html.Div([
                    html.Label("Worm:", style = {'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                    dcc.Dropdown(
                        id='dropdown-worm',
                        options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                        value=0, 
                        clearable=False
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Behaviour:", style = {'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                    dcc.Dropdown(
                        id='dropdown-behavior',
                        options=behavior_options,
                        value='ALL', 
                        clearable=False
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),

                html.Div([
                html.Label("Neuron type:", style = {'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
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
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '2%'})

            ], style={'padding': '20px', 'backgroundColor': "#9c9c9c", 'marginBottom': '20px', 'borderRadius': '5px'}),
            
            # PLOTS
            html.Div([
                #  LEFT COLUMN
                html.Div([
                    html.Img(id='global-beeswarm', style={'maxWidth': '100%', 'display': 'block', 'margin': 'auto'})
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # RIGHT COLUMN
                html.Div([
                    dcc.Graph(id='global-bar-plot'),
                    html.Div(dcc.Graph(id='global-table'), style={'marginTop': '0px'})
                ], style={'width': '58%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
                
            ], style={'width': '100%'}),

            html.Div([
                html.H3("Quality check results", style={'textAlign': 'center', 'color': "#F4F6F7", 'marginTop': '0px', 'marginBottom': '20px'}),
                
                html.Div([
                    # RAR RESULTS
                    html.Div([
                        html.H4("Test RAR (SHAP)", style={'color': '#7f8c8d'}),
                        html.Div(id='quality-rar-result-shap', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': "#F4F6F7"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),

                    html.Div([
                        html.H4("Test RAR (IG)", style={'color': '#7f8c8d'}),
                        html.Div(id='quality-rar-result-ig', style={'fontSize': '28px', 'fontWeight': 'bold', 'color':"#F4F6F7"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                    
                    # CORRELATION RESULTS
                    html.Div([
                        html.H4("Rankings correlation (spearman)", style={'color': '#7f8c8d'}),
                        html.Div(id='quality-corr-result', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': "#F4F6F7"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),

                    html.Div([
                        html.H4("P.value for correlation", style={'color': '#7f8c8d'}),
                        html.Div(id='quality-pval-result', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': "#F4F6F7"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'})
                ])
                
            ], style={
                'width': '100%', 'marginTop': '40px', 'padding': '20px', 
                'backgroundColor': "#166ea9", 'borderRadius': '10px', 
                'boxShadow': '0px 2px 4px rgba(0,0,0,0.1)'
            })
        ]), "Global Panel"

    elif pathname == "/worm":
            fig_xai_empty = empty_xai_bar_plots()
    
            
            return html.Div([

                 # FILTERING PANEL
                html.Div([

                    html.Div([
                        html.Label("Worm:", style = {'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='dropdown-worm',
                            options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                            value=0, 
                            clearable=False
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    
                    html.Div([
                    html.Label("Neuron type:", style = {'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
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
                    ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '2%'}),

                    html.Div([
                        html.Label('Time Window:', style={'fontWeight': 'bold', 'color': '#F7F0F8', 'fontSize': '16px'}),
                        dbc.InputGroup([
                            dbc.Button("◄", id="btn-time-prev", n_clicks=0, color="secondary", outline=True),
                            dbc.Input(id="input-time", type="number", value=0, min=0, step=1, style={'textAlign': 'center'}),
                            dbc.Button("►", id="btn-time-next", n_clicks=0, color="secondary", outline=True)
                        ], style={'width': '150px'})
                    ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
    
                ], style={'padding': '20px', 'backgroundColor': "#9c9c9c", 'marginBottom': '20px', 'borderRadius': '5px'}),

                # PLOTS           
                html.Div(dcc.Graph(id='plot-3d'), style={'width': '48%', 'display': 'inline-block'}),
                html.Div(dcc.Graph(id='plot-xai', figure=fig_xai_empty), style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.H3(id='quality-behavior-title', style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '20px'}),
                    
                    dash_table.DataTable(
                        id='quality-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial'},
                        style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
                    )
                ], style={'width': '100%', 'marginTop': '40px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'boxShadow': '0px 2px 4px rgba(0,0,0,0.1)'})
            ]), "Worm Specific Panel"
        
    elif pathname == "/temp":
        fig = temporal_dashboard(x_flat, b_, behavior_names, shap_matrix, ig_matrix, neuron_array, palette, title= "Thermal dashboard")
        return html.Div([dcc.Graph(figure=fig)]), "Temporal Plot"

    elif pathname == "/quality":
        return html.Div([
            html.Div([
            
                    html.Div([
                        html.Label("Worm:", style = {'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                        dcc.Dropdown(
                            id='dropdown-worm',
                            options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                            value=0, 
                            clearable=False
                        )
                    ], style={'width': '30%', 'display': 'left'}),
                    
                ], style={'padding': '20px', 'backgroundColor': "#9c9c9c", 'marginBottom': '20px', 'borderRadius': '5px'}),

            html.Div(id='quality-summary-container', style={'marginTop': '30px'})

        ]), "Quality check summary"
        
    

############ GLOBAL PANEL ########################################################

@app.callback(
    [Output('global-bar-plot', 'figure'),
     Output('global-table', 'figure'),
     Output('global-beeswarm', 'src'),
     Output('quality-rar-result-shap', 'children'),  
     Output('quality-rar-result-ig', 'children'),  
     Output('quality-corr-result', 'children'),
     Output('quality-pval-result', 'children')],
    [Input('dropdown-worm', 'value'),
     Input('dropdown-behavior', 'value'),
     Input('dropdown-neuron-type', 'value')]
)
def update_global_dashboard(selected_worm, selected_behavior, selected_neuron_type):

    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=selected_worm)
    target_b = None if selected_behavior == 'ALL' else selected_behavior

    w_key = str(selected_worm)
    b_key = str(selected_behavior)

    if w_key in quality_metrics and b_key in quality_metrics[w_key]:

        rar_val_shap = quality_metrics[w_key][b_key].get('rar_shap', 'Empty')
        rar_val_ig = quality_metrics[w_key][b_key].get('rar_ig', 'Empty')
        corr_val = quality_metrics[w_key][b_key].get('corr', 'Empty')
        p_val = quality_metrics[w_key][b_key].get('corr_p_value', "Empty")

        rar_text_shap = f"{rar_val_shap:.3f}" if isinstance(rar_val_shap, (int, float)) else str(rar_val_shap)
        rar_text_ig = f"{rar_val_ig:.3f}" if isinstance(rar_val_ig, (int, float)) else str(rar_val_ig)
        corr_text = f"{corr_val:.3f}" if isinstance(corr_val, (int, float)) else str(corr_val)
        p_val = f"{p_val:.3f}" if isinstance(p_val, (int, float)) else str(p_val)

    else:
        rar_text_shap = "No data"
        rar_text_ig = "No data"
        corr_text = "No data"
        p_val = "No data"


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
    
    return fig_bars, fig_table, beeswarm_src, rar_text_shap,rar_text_ig, corr_text, p_val


################# WORM SPECIFIC PANEL ####################################################

@app.callback(
    Output('input-time', 'value'),
    [Input('btn-time-prev', 'n_clicks'),
     Input('btn-time-next', 'n_clicks'),
     Input('plot-3d', 'clickData')],
    State('input-time', 'value'),
    prevent_initial_call=True
)
def update_time_state(prev_clicks, next_clicks, clickData, current_time):

    if current_time is None:
        current_time = 0

    triggered_id = ctx.triggered_id

    if triggered_id == 'btn-time-prev':
        return max(0, current_time - 1) 

    elif triggered_id == 'btn-time-next':
        return current_time + 1

    elif triggered_id == 'plot-3d' and clickData is not None:
        point_data = clickData['points'][0]
        text_data = point_data.get('text', '')
        try:
            time_part = text_data.split('<br>')[0]
            time_idx_str = time_part.split(':')[1].strip()
            return int(time_idx_str)
        except:
            return point_data.get('pointNumber', current_time)

    return current_time

@app.callback(
    Output('plot-3d', 'figure'),
    [Input('dropdown-worm', 'value'),
     Input('input-time', 'value')] 
)
def update_worm_specific_dashboard(selected_worm, time_idx):

    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=selected_worm)

    fig_3d = latent_space_3D(selected_worm, x_flat, model ,b_,  behavior_names, palette)

    fig_3d.update_layout(uirevision=str(selected_worm))

    if time_idx is not None and time_idx < len(x_flat):
        import torch
        import plotly.graph_objects as go
        device = next(model.parameters()).device

        x_tensor = torch.tensor(x_flat[time_idx:time_idx+1], dtype=torch.float32).to(device)
        with torch.no_grad():
            point_3d = model.tau(x_tensor).cpu().numpy()[0]

        fig_3d.add_trace(go.Scatter3d(
            x=[point_3d[0]], 
            y=[point_3d[1]], 
            z=[point_3d[2]],
            mode='markers',
            marker=dict(size=12, color='#2c3e50', symbol='circle', line=dict(color='red', width=2)),
            name='Wybrana klatka',
            showlegend=False,
            hoverinfo='skip' 
        ))


    return fig_3d




@app.callback(
    Output('plot-xai', 'figure'),
    [Input('input-time', 'value'),             
     Input('dropdown-neuron-type', 'value')],
    State('dropdown-worm', 'value'),
    prevent_initial_call=True
)
def update_xai_on_time_change(time_idx, selected_neuron_type, selected_worm):
    if time_idx is None:
        return dash.no_update

    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=selected_worm)

    if time_idx >= len(b_):
        time_idx = len(b_) - 1

    current_shap = shap_matrix[time_idx]
    current_ig = ig_matrix[time_idx]
    
    shap_margin = (np.max(current_shap) - np.min(current_shap)) * 0.05
    ig_margin = (np.max(current_ig) - np.min(current_ig)) * 0.05
    
    shap_y_range = [np.min(current_shap) - shap_margin, np.max(current_shap) + shap_margin]
    ig_y_range = [np.min(current_ig) - ig_margin, np.max(current_ig) + ig_margin]

    if selected_neuron_type != 'ALL':
        valid_indices = [
            i for i, name in enumerate(neuron_array) 
            if neuron_types_dict.get(name, 'Unidentified') == selected_neuron_type
        ]
        
        if len(valid_indices) == 0:
            import plotly.graph_objects as go
            return go.Figure().update_layout(title=f"No neurons in the type: {selected_neuron_type}")

        neuron_array = neuron_array[valid_indices]
        shap_matrix = shap_matrix[:, valid_indices]
        ig_matrix = ig_matrix[:, valid_indices]


    return CLICK_xai_barplots(time_idx, shap_matrix, ig_matrix, neuron_array, b_, behavior_names, shap_y_range, ig_y_range)


@app.callback(
    [Output('quality-behavior-title', 'children'),
     Output('quality-table', 'data'),
     Output('quality-table', 'columns')],
    [Input('input-time', 'value'),
     Input('dropdown-worm', 'value')]
)
def update_inter_worm_panel(time_idx, selected_worm):
    if time_idx is None:
        time_idx = 0

    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=selected_worm)
    
    if time_idx >= len(b_):
        time_idx = len(b_) - 1
        
    current_behavior_idx = b_[time_idx]
    current_behavior_name = behavior_names.get(current_behavior_idx, "Unknown behaviour")

    worm_str = str(selected_worm)
    beh_str = str(current_behavior_idx)
    
    metrics = quality_metrics.get(worm_str, {}).get(beh_str, {})
    
    if not metrics:
        return f"Inter-worm Correlation: {current_behavior_name} (No data in json)", [], []


    table_dict = metrics.get('inter_worm_table', {})
    table_data = table_dict.get('data', [])
    
    if table_data:
        table_columns = [{"name": col, "id": col} for col in table_data[0].keys()]
    else:
        table_columns = []

    title = f"Inter-worm Correlation for current behaviour: {current_behavior_name}"

    return title, table_data, table_columns

##################### QUALITY CHECK SUMMARY ########################################################

@app.callback(
    Output('quality-summary-container', 'children'),
    Input('dropdown-worm', 'value')
)
def update_quality_summary_panel(selected_worm):


    worm_str = str(selected_worm)
    worm_data = quality_metrics.get(worm_str, {})
    
    if not worm_data:
        return html.Div(f"Brak danych dla Robaka {selected_worm}")

    global_metrics = worm_data.get('ALL', {})
    
    global_section = html.Div([
        html.H3("Global Quality Metrics (Całe nagranie)", style={'textAlign': 'center', 'color': '#2C3E50'}),
        html.Div([
            html.Div([html.H4("RAR (SHAP)"), html.Div(global_metrics.get('rar_shap', 'Brak'), style={'fontSize': '24px'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
            html.Div([html.H4("RAR (IG)"), html.Div(global_metrics.get('rar_ig', 'Brak'), style={'fontSize': '24px'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
            html.Div([html.H4("Korelacja Rankingów"), html.Div(global_metrics.get('corr', 'Brak'), style={'fontSize': '24px', 'color': '#27ae60'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
            html.Div([html.H4("P-value"), html.Div(global_metrics.get('corr_p_value', 'Brak'), style={'fontSize': '24px'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
        ], style={'marginBottom': '40px', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px'})
    ])


    tables_elements = []
    
    for beh_idx, beh_data in worm_data.items():
        if beh_idx == 'ALL':
            continue 
            
        table_info = beh_data.get('inter_worm_table', {})
        table_title = table_info.get('title', f"Zachowanie {beh_idx}")
        table_records = table_info.get('data', [])
 
        tables_elements.append(html.H4(table_title, style={'marginTop': '30px', 'color': '#34495e'}))
        
        if table_records:
            table_columns = [{"name": col, "id": col} for col in table_records[0].keys()]

            dt = dash_table.DataTable(
                data=table_records,
                columns=table_columns,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
            )
            tables_elements.append(dt)
        else:
            tables_elements.append(html.P("Brak wystarczających danych do korelacji dla tego zachowania.", style={'color': '#7f8c8d'}))

    final_layout = html.Div([global_section] + tables_elements)
    
    return final_layout









########### APPLICATION CALLING ################################

if __name__ == '__main__':
    app.run(debug=True)