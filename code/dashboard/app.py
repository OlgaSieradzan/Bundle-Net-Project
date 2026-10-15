# Paths
import os
import sys
import numpy as np
import json
from pathlib import Path


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Dodajemy go do środowiska Pythona
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Libraries

import dash
from data_loader import load_all_data
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, ctx, dash_table
from panel_global_importance import global_importance_barplot, comparison_table_xai, shap_beeswarm
from panel_heatmap import temporal_dashboard
from panel_worm_specific import latent_space_3D, empty_xai_bar_plots, CLICK_xai_barplots
from xai_methods import BehaviorWrapper
from panel_corr import inter_worm_corr, behaviour_corr, neurons_corr


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



def create_knowledge_card(title, description, links):

    links_elements = [
        html.A(
            link_text, 
            href=link_url, 
            target="_blank", 
            style={
                'display': 'inline-block',
                'backgroundColor': '#2980b9',
                'color': 'white',
                'padding': '8px 15px',
                'marginRight': '10px',
                'marginBottom': '10px',
                'borderRadius': '5px',
                'textDecoration': 'none',
                'fontSize': '14px',
                'fontWeight': 'bold',
                'boxShadow': '0px 2px 4px rgba(0,0,0,0.1)'
            }
        ) for link_text, link_url in links
    ]

    return html.Div([
        html.H3(title, style={'fontWeight': 'bold', 'color': '#2c3e50', 'borderBottom': '2px solid #bdc3c7', 'paddingBottom': '10px'}),
        html.P(description, style={'fontSize': '16px', 'color': '#34495e', 'lineHeight': '1.6', 'textAlign': 'justify'}),
        html.Div(links_elements, style={'marginTop': '15px'})
    ], style={
        'backgroundColor': 'white', 
        'padding': '25px', 
        'marginBottom': '25px', 
        'borderRadius': '8px', 
        'boxShadow': '0px 4px 6px rgba(0,0,0,0.05)'
    })


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
                dbc.NavLink("Correlation Panel", href="/corr", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'}),
                dbc.NavLink("Quality check summary", href="/quality", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'}),
                dbc.NavLink("Knowledge section", href="/knowledge", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'}),
                dbc.NavLink("Download data", href= "/download", active="exact", style={'fontSize': '18px', 'marginBottom': '10px'})
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
                    html.Label("Worm:", style={'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                    dcc.Dropdown(
                        id='dropdown-worm',
                        options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                        value=0, 
                        clearable=False
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Behaviour:", style={'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                    dcc.Dropdown(
                        id='dropdown-behavior',
                        options=behavior_options,
                        value='ALL', 
                        clearable=False
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),

                html.Div([
                    html.Label("Neuron type:", style={'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
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

            # QUALITY CHECK RESULTS
            html.Div([

                html.H3("Quality check results", style={'textAlign': 'center', 'color': "#166ea9", 'marginTop': '40px', 'fontWeight': 'bold'}),
                
                html.Div([
                    # RAR RESULTS
                    html.Div([
                        html.H4("Model's accuracy after RAR (SHAP)"),
                        html.Div(id='quality-rar-result-shap', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': "white"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),

                    html.Div([
                        html.H4("Model's accuracy after RAR (IG):"),
                        html.Div(id='quality-rar-result-ig', style={'fontSize': '28px', 'fontWeight': 'bold', 'color':"white"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                    
                    # CORRELATION RESULTS
                    html.Div([
                        html.H4("IG and SHAP Ranking correlation:"),
                        html.Div(id='quality-corr-result', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': "white"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),

                    html.Div([
                        html.H4("P-value for correlation:"),
                        html.Div(id='quality-pval-result', style={'fontSize': '28px', 'fontWeight': 'bold', 'color': "white"})
                    ], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'})
                ])

            ])
            

        ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': "#e0e0e0", 'borderRadius': '10px'}), "Global Panel"

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
                    html.H3(id='quality-behavior-title', style={'textAlign': 'center', 'color':  "#166ea9", 'marginBottom': '20px'}),
                    
                    dash_table.DataTable(
                        id='quality-table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial'},
                        style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
                    )
                ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': "#e0e0e0", 'borderRadius': '10px'})
            ]), "Worm Specific Panel"
        
    elif pathname == "/temp":

        return html.Div([
        html.Div([
            html.Div([
                html.Label("Worm:", style={'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                dcc.Dropdown(
                    id='temporal-dropdown-worm', 
                    options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                    value=0, 
                    clearable=False,
                    style={'color': 'black'}
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Neuron type:", style={'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                dcc.Dropdown(
                    id='temporal-dropdown-neuron',
                    options=[
                        {'label': 'All', 'value': 'ALL'},
                        {'label': 'Sensory', 'value': 'Sensory'},
                        {'label': 'Motor', 'value': 'Motor'},
                        {'label': 'Interneuron', 'value': 'Interneuron'},
                        {'label': 'Unidentified', 'value': 'Unidentified'}
                    ],
                    value='ALL', 
                    clearable=False,
                    style={'color': 'black'}
                )

            
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '2%'}),

            html.Div([
                html.Label("Top 8 Ranking by:", style={'fontWeight': 'bold', 'color': "#F7F0F8", 'fontSize': '16px'}),
                dcc.Dropdown(
                    id='temporal-dropdown-ranking',
                    options=[], 
                    value='ALL_COMBINED', 
                    clearable=False, 
                    style={'color': 'black'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '2%'}),
            
        ], style={'padding': '20px', 'backgroundColor': "#9c9c9c", 'marginBottom': '20px', 'borderRadius': '5px'}),
        
        
        html.Div([
            dcc.Graph(id='temporal-plot', style={'height': '1100px'}) 
        ])
    ], style={'padding': '20px'}), "Heatmap panel"

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

    elif pathname == "/corr":
        return html.Div([
                
        dcc.Tabs(id="tabs-correlation", value='tab-inter-worm', children=[
            
            
            # TAB 1: Inter-worm correlation
            
            dcc.Tab(label='Inter-Worm Correlation', value='tab-inter-worm', style={'fontWeight': 'bold'}, selected_style={'backgroundColor': '#166ea9', 'color': 'white', 'fontWeight': 'bold'}, children=[
                html.Div([
                    html.Div([
                        html.Label("Behaviour:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-worm-behavior',
                            options=behavior_options, 
                            value='ALL', clearable=False
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Metric:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-worm-metric',
                            options=[
                                {'label': 'Raw Activity', 'value': 'activity'},
                                {'label': 'SHAP', 'value': 'shap'},
                                {'label': 'Integrated Gradients', 'value': 'ig'}
                            ],
                            value='shap', clearable=False
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '2%'})
                ], style={'padding': '20px', 'backgroundColor': "#9c9c9c", 'margin': '20px 0', 'borderRadius': '5px'}),
                
                
                dcc.Graph(id='plot-corr-inter-worm', style={'display': 'flex', 'justifyContent': 'center'})
            ]),

            
            # TAB 2: Behavior Rankings Correlation
            
            dcc.Tab(label='Behavior Rankings (XAI)', value='tab-behavior', style={'fontWeight': 'bold'}, selected_style={'backgroundColor': '#166ea9', 'color': 'white', 'fontWeight': 'bold'}, children=[
                html.Div([
                    html.Div([
                        html.Label("Worm:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-beh-worm',
                            options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                            value=0, clearable=False
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("XAI Method:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-beh-metric',
                            options=[
                                {'label': 'Raw Activity', 'value': 'activity'},
                                {'label': 'SHAP', 'value': 'shap'},
                                {'label': 'Integrated Gradients', 'value': 'ig'}
                            ],
                            value='shap', clearable=False
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '2%'})
                ], style={'padding': '20px', 'backgroundColor': "#9c9c9c", 'margin': '20px 0', 'borderRadius': '5px'}),
                
                
                dcc.Graph(id='plot-corr-behavior', style={'height': '700px'})
            ]),

            
            # TAB 3: Neuron-to-Neuron Correlation
            
            dcc.Tab(label='Neuron Network Correlation', value='tab-neuron', style={'fontWeight': 'bold'}, selected_style={'backgroundColor': '#166ea9', 'color': 'white', 'fontWeight': 'bold'}, children=[
                html.Div([
                    html.Div([
                        html.Label("Worm:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-neuron-worm',
                            options=[{'label': f'Worm {i}', 'value': i} for i in range(5)], 
                            value=0, clearable=False
                        )
                    ], style={'width': '18%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Behaviour:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-neuron-behavior',
                            options=behavior_options,
                            value='ALL', clearable=False
                        )
                    ], style={'width': '18%', 'display': 'inline-block', 'marginLeft': '2%'}),
                    
                    html.Div([
                        html.Label("Metric:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-neuron-metric',
                            options=[
                                {'label': 'Raw Activity', 'value': 'activity'},
                                {'label': 'SHAP', 'value': 'shap'},
                                {'label': 'Integrated Gradients', 'value': 'ig'}
                            ],
                            value='activity', clearable=False
                        )
                    ], style={'width': '18%', 'display': 'inline-block', 'marginLeft': '2%'}),

                    html.Div([
                        html.Label("Neuron type:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-neuron-type',
                            options=[
                                {'label': 'All', 'value': 'ALL'},
                                {'label': 'Sensory', 'value': 'Sensory'},
                                {'label': 'Motor', 'value': 'Motor'},
                                {'label': 'Interneuron', 'value': 'Interneuron'},
                                {'label': 'Unidentified', 'value': 'Unidentified'}
                            ],
                            value='ALL', clearable=False
                        )
                    ], style={'width': '18%', 'display': 'inline-block', 'marginLeft': '2%'}),

                    html.Div([
                        html.Label("Specific Neurons:", style={'fontWeight': 'bold', 'color': "#F7F0F8"}),
                        dcc.Dropdown(
                            id='corr-neuron-names',
                            options=[], # Puste - doczytywane dynamicznie
                            value=[],   # Pusta lista domyślnie
                            multi=True, # Pozwala wybrać wiele nazw z listy
                            placeholder="Select specific neurons..."
                        )
                    ], style={'width': '18%', 'display': 'inline-block', 'marginLeft': '2%'})

                ], style={'padding': '20px', 'backgroundColor': "#9c9c9c", 'margin': '20px 0', 'borderRadius': '5px'}),
                
                
                dcc.Graph(id='plot-corr-neuron', style={'height': '4000px'}) 
            ])
        ])
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'}), "Correlation Panel"



    elif pathname == "/knowledge":
        return html.Div([
        
        # Nagłówek sekcji
            html.Div([
                html.H1("Knowledge Base", style={'fontWeight': 'bold', 'color': '#2c3e50', 'textAlign': 'center'}),
                html.P("In this section, there are short descriptions needed to understand the analysis" \
                " presented in the dashboard. For a broader understanding of the subject, each part includes" \
                " additional links with articles or websites explaining the concepts in detail.", 
                    style={'fontSize': '18px', 'color': '#7f8c8d', 'textAlign': 'center', 'maxWidth': '800px', 'margin': '0 auto 40px auto'})
            ]),

            # Karty z wiedzą
            create_knowledge_card(
                title="Explainable AI (XAI)",
                description="Explainable Artificial Intelligence, also referred to as XAI,"
                " is a group of methods and frameworks designed to help understand the decision-making " \
                "process behind complex machine learning algorithms. It bridges the gap between 'black-box' models and human interpretability.",
                links=[
                    ("IBM: What is XAI?", "https://www.ibm.com/topics/explainable-ai"),
                    ("Interpretable ML Book", "https://christophm.github.io/interpretable-ml-book/"),
                    ("XAI in Joint modelling", "https://www.nature.com/articles/s41583-025-00996-1"),
                    ("XAI in Medicine", "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11071277")
                ]
            ),

            create_knowledge_card(
                title="Shapley Additive exPlanations (SHAP)",
                description="SHAP is a model-agnostic approach that provides both local and" \
                " global explanations. It tells us how much each input (feature) is helping or" \
                " hurting the final prediction. The main idea is derived from game theory: it fairly" \
                " distributes the 'payout' (the prediction) among all features based on their marginal contribution.",
                links=[
                    ("Original NIPS Paper", "https://arxiv.org/abs/1705.07874"),
                    ("SHAP Python Docs", "https://shap.readthedocs.io/en/latest/"),
                    ("SHAP usage example with Time series data", "https://link.springer.com/article/10.1186/s10194-026-02320-3"),
                    ("SHAP vs other post-hoc xai methods", "https://arxiv.org/abs/2605.28977")
                ]
            ),

            create_knowledge_card(
                title="Integrated Gradients (IG)",
                description="Integrated Gradients is a global approach based on calculating attribution with gradients"
                ". It aggregates the gradients along the inputs that fall on the straight line between a baseline (e.g., zero signal)"
                " and the actual input, ensuring sensitivity and implementation invariance.",
                links=[
                    ("Original ICML Paper", "https://arxiv.org/abs/1703.01365"),
                    ("Captum (PyTorch) IG", "https://captum.ai/docs/algorithms#integrated-gradients"),
                    ("TensorFlow IG Tutorial", "https://www.tensorflow.org/tutorials/interpretability/integrated_gradients"),
                    ("IG and SGIG usage example on Time series data", "https://www.nature.com/articles/s41598-022-15539-2")
                ]
            ),

            create_knowledge_card(
                title="Retain and Retrain Test (RAR)",
                description="An approach for testing the faithfulness of XAI methods. The idea is that if" \
                " an XAI method correctly identifies the most important neurons, then after retaining only" \
                " the top n% of these neurons and retraining the model, the accuracy shouldn't drop significantly" \
                " compared to random selection. In this dashboard, 10% of top neurons were retained and the after retraing accuracy was reported.",
                links=[
                    ("RaR example usage", "https://link.springer.com/article/10.1186/s10194-026-02320-3") 
                ]
            ),

            create_knowledge_card(
                title="Spearman Correlation",
                description="Spearman’s Rank Correlation is a nonparametric statistical measure used to find" \
                " the strength and direction of association between two ranked variables. It checks how well" \
                " the relationship between two variables can be described using a monotonic function, making it ideal for comparing ranking outputs from SHAP and IG.",
                links=[
                    ("Understanding Spearman's Rank", "https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php"),
                    ("SciPy Documentation", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"),
                    ("StatQuest: Correlation", "https://www.youtube.com/watch?v=qtaqvPAeEJY")
                ]
            ),

            create_knowledge_card(
                title="Latent Space",
                description="Latent space is a mathematical space where complex, high-dimensional data is represented"
                " in a compressed, lower-dimensional format. In this dashboard, the 3D latent space visualizes how the" \
                " neural network groups similar worm behaviors together, mapping biological activity into spatial distances.",
                links=[
                    ("Understanding Latent Space", "https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d"),
                    ("Dimensionality Reduction", "https://scikit-learn.org/stable/modules/decomposition.html"),
                    ("Autoencoders & Latent Spaces", "https://www.jeremyjordan.me/autoencoders/")
                ]
            ),

            create_knowledge_card(
                title="Raw Neurons Activation",
                description="Raw neuronal activation refers to the direct, unprocessed signals recorded from individual" \
                " neurons over time. In C. elegans research, this is typically measured via calcium imaging (fluorescence intensity)," \
                " showing exactly when and how strongly a specific neuron fires during a behavioral sequence.",
                links=[
                    ("Calcium Imaging Overview", "https://en.wikipedia.org/wiki/Calcium_imaging"),
                    ("C. elegans Connectome", "https://www.wormatlas.org/"),
                    ("Neural Decoding", "https://en.wikipedia.org/wiki/Neural_decoding")
                ]
            )

    ], style={'padding': '40px', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'}), "Knowledge Base"

    elif pathname == "/download":
        return html.Div([
        html.H2("Export Analysis Data", style={'fontWeight': 'bold', 'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '40px'}),
        
        html.Div([
            html.P("Download the complete analysis dataset including all XAI rankings (SHAP, IG), Retain-and-Retrain results, and calculated correlation matrices. The data will be formatted into a multi-sheet Microsoft Excel file.", 
                   style={'fontSize': '18px', 'textAlign': 'center', 'marginBottom': '40px', 'color': '#34495e'}),
            
            
            html.Button(
                "Download Full Analysis (Excel)", 
                id="btn-download-excel", 
                style={
                    'backgroundColor': '#27ae60', 'color': 'white', 
                    'fontSize': '20px', 'fontWeight': 'bold',
                    'padding': '15px 30px', 'borderRadius': '8px',
                    'border': 'none', 'cursor': 'pointer',
                    'display': 'block', 'margin': '0 auto',
                    'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)'
                }
            ),
            
            # Ukryty komponent odpowiedzialny za triggerowanie pobierania u klienta
            dcc.Download(id="download-dataframe-xlsx")
            
        ], style={'backgroundColor': 'white', 'padding': '50px', 'borderRadius': '10px', 'boxShadow': '0px 4px 15px rgba(0,0,0,0.05)', 'maxWidth': '800px', 'margin': '0 auto'}),

        html.H2("Detailed decription of data", style={'fontSize': '18px', 'textAlign': 'center', 'marginBottom': '40px', 'color': '#34495e'}),
        html.Ul([
            html.Li("Sheet I - ranking data"),
            html.Li("Drugi element")
        ])
        
    ], style={'padding': '40px', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'}), "Download Data"
        
    

############ GLOBAL PANEL CALLBACKS ########################################################

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


################# WORM SPECIFIC PANEL CALLBACKS ####################################################

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

##################### HEATMAP CALLBACKS #####################################################################

@app.callback(
    Output('temporal-dropdown-ranking', 'options'),
    Input('temporal-dropdown-worm', 'value')
)
def update_ranking_options(selected_worm):
    _, _, _, _, behavior_names, _, _, _ = load_all_data(worm_idx=selected_worm)

    options = [
        {'label': 'Overall Combined (SHAP + IG)', 'value': 'ALL_COMBINED'},
        {'label': 'Top Overall SHAP', 'value': 'ALL_SHAP'},
        {'label': 'Top Overall IG', 'value': 'ALL_IG'}
    ]

    for idx, name in behavior_names.items():
        options.append({'label': f'Top SHAP: {name}', 'value': f'SHAP_{idx}'})
        options.append({'label': f'Top IG: {name}', 'value': f'IG_{idx}'})
        
    return options



@app.callback(
    Output('temporal-plot', 'figure'),
    [Input('temporal-dropdown-worm', 'value'),
     Input('temporal-dropdown-neuron', 'value'),
     Input('temporal-dropdown-ranking', 'value')]
)
def update_temporal_panel(selected_worm, selected_neuron_type, selected_ranking):
    
    x_flat, b_, shap_matrix, ig_matrix, behavior_names, neuron_array, palette, model = load_all_data(worm_idx=selected_worm)

    global_x_min = np.min(x_flat)
    global_x_max = np.max(x_flat)
    global_shap_max = np.max(np.abs(shap_matrix))
    global_ig_max = np.max(np.abs(ig_matrix))

    valid_indices = np.arange(len(neuron_array))
    if selected_neuron_type != 'ALL':
        valid_indices = [
            i for i, name in enumerate(neuron_array) 
            if neuron_types_dict.get(name, 'Unidentified') == selected_neuron_type
        ]
        if len(valid_indices) == 0:
            import plotly.graph_objects as go
            return go.Figure().update_layout(title=f"Brak neuronów typu: {selected_neuron_type}")
    
    neuron_array = neuron_array[valid_indices]
    x_flat = x_flat[:, valid_indices]
    shap_matrix = shap_matrix[:, valid_indices]
    ig_matrix = ig_matrix[:, valid_indices]

    if selected_ranking:
        if selected_ranking == 'ALL_COMBINED':
            mean_s = np.mean(np.abs(shap_matrix), axis=0)
            mean_i = np.mean(np.abs(ig_matrix), axis=0)
            scores = (mean_s / (np.max(mean_s) + 1e-8)) + (mean_i / (np.max(mean_i) + 1e-8))
        elif selected_ranking == 'ALL_SHAP':
            scores = np.mean(np.abs(shap_matrix), axis=0)
        elif selected_ranking == 'ALL_IG':
            scores = np.mean(np.abs(ig_matrix), axis=0)
        else:
            method, beh_idx_str = selected_ranking.split('_')
            beh_idx = int(beh_idx_str)

            mask = (b_.flatten() == beh_idx)
            
            if not np.any(mask):
                scores = np.zeros(len(neuron_array))
            else:
                matrix_to_use = shap_matrix if method == 'SHAP' else ig_matrix
                scores = np.mean(np.abs(matrix_to_use[mask]), axis=0)

        TOP_N = min(8, len(neuron_array))
        top_indices = np.argsort(scores)[-TOP_N:] 
        
        neuron_array = neuron_array[top_indices]
        x_flat = x_flat[:, top_indices]
        shap_matrix = shap_matrix[:, top_indices]
        ig_matrix = ig_matrix[:, top_indices]

    title = f"Temporal Analysis - Worm {selected_worm} ({selected_neuron_type} Neurons)"
    
    fig = temporal_dashboard(
        x_raw=x_flat, 
        b_raw=b_, 
        behavior_names=behavior_names, 
        shap_matrix=shap_matrix, 
        ig_matrix=ig_matrix, 
        neuron_names=neuron_array, 
        palette=palette, 
        title=title, 
        top_n=8,
        global_x_range=[global_x_min, global_x_max], 
        global_shap_max=global_shap_max,
        global_ig_max=global_ig_max
    )

    fig.update_layout(uirevision=f"{selected_worm}_{selected_neuron_type}_{selected_ranking}")
    
    return fig

##################### CORRELATION CALLBACKS ###########################################################

@app.callback(
    Output('corr-neuron-names', 'options'),
    [Input('corr-neuron-worm', 'value'),
     Input('corr-neuron-type', 'value')] 
)
def update_neuron_names_options(selected_worm, selected_type):
    _, _, _, _, _, neuron_array, _, _ = load_all_data(worm_idx=selected_worm)
    

    if selected_type != 'ALL':
        valid_neurons = [n for n in neuron_array if neuron_types_dict.get(n, 'Unidentified') == selected_type]
    else:
        valid_neurons = neuron_array
        
    return [{'label': n, 'value': n} for n in valid_neurons]

@app.callback(
        Output('plot-corr-inter-worm', 'figure'),
        [Input('corr-worm-behavior', 'value'),
        Input('corr-worm-metric', 'value')
        ]
)
def update_inter_worm_corr(selected_behaviour, selected_metric):

    return inter_worm_corr(selected_behaviour, selected_metric)


@app.callback(
        Output('plot-corr-behavior', 'figure'),
        [Input('corr-beh-worm', 'value'),
        Input('corr-beh-metric', 'value')
        ]
)
def update_inter_worm_corr(target_worm, target_metric):

    return behaviour_corr(target_worm, target_metric)



@app.callback(
    Output('plot-corr-neuron', 'figure'),
    [Input('corr-neuron-behavior', 'value'),
     Input('corr-neuron-worm', 'value'),
     Input('corr-neuron-metric', 'value'),
     Input('corr-neuron-type', 'value'), # Dodane
     Input('corr-neuron-names', 'value')] 
)
def update_neuron_corr_panel(selected_behavior, selected_worm, selected_metric, selected_type, selected_names):
    fig = neurons_corr(
        target_behaviour=selected_behavior,
        target_worm=selected_worm,
        target_metric=selected_metric,
        filtered_neuron_type=selected_type, 
        filtered_neuron_names=selected_names, 
        neuron_types_dict=neuron_types_dict
    )
    return fig

##################### QUALITY CHECK SUMMARY CALLBACKS########################################################

@app.callback(
    Output('quality-summary-container', 'children'),
    Input('dropdown-worm', 'value')
)
def update_quality_summary_panel(selected_worm):

    worm_str = str(selected_worm)
    worm_data = quality_metrics.get(worm_str, {})
    
    if not worm_data:
        return html.Div(f"No data for worm: {selected_worm}")

    elements = []
    
    for beh_idx, beh_data in worm_data.items():
        # 1. Określenie tytułu dla sekcji
        if beh_idx == 'ALL':
            section_title = "Global Quality Metrics (Whole recording)"
        else:
            table_info = beh_data.get('inter_worm_table', {})
            section_title = table_info.get('title', f"Behaviour: {beh_idx}")
            
        # 2. Generowanie kafelków z metrykami (RAR i korelacja) dla danej sekcji
        metrics_section = html.Div([
            html.H3(section_title, style={'textAlign': 'center', 'color': "#166ea9", 'marginTop': '40px', 'fontWeight': 'bold'}),
            html.Div([
                html.Div([html.H4("Model's accuracy after RAR (SHAP):", style={'fontSize': '16px'}), html.Div(beh_data.get('rar_shap', 'Brak'), style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([html.H4("Model's accuracy after RAR (IG):", style={'fontSize': '16px'}), html.Div(beh_data.get('rar_ig', 'Brak'), style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([html.H4("IG and SHAP Ranking correlation:", style={'fontSize': '16px'}), html.Div(beh_data.get('corr', 'Brak'), style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
                html.Div([html.H4("P-value for correlation:", style={'fontSize': '16px'}), html.Div(beh_data.get('corr_p_value', 'Brak'), style={'fontSize': '24px', 'fontWeight': 'bold'})], style={'width': '24%', 'display': 'inline-block', 'textAlign': 'center'}),
            ], style={'marginBottom': '20px', 'padding': '20px', 'backgroundColor': "#e0e0e0", 'borderRadius': '10px'}) # Możesz dostosować kolor tła (np. #e0e0e0)
        ])
        
        elements.append(metrics_section)
        
        # 3. Generowanie tabeli (Inter-worm) dla danej sekcji
        table_info = beh_data.get('inter_worm_table', {})
        table_records = table_info.get('data', [])
        
        if table_records:
            table_columns = [{"name": col, "id": col} for col in table_records[0].keys()]
            dt = dash_table.DataTable(
                data=table_records,
                columns=table_columns,
                style_table={'overflowX': 'auto', 'marginBottom': '40px'}, # Odstęp na dole tabeli
                style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial'},
                style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
            )
            elements.append(dt)
        else:
            elements.append(html.P("Not enough data for inter-worm correlation.", style={'color': '#7f8c8d', 'textAlign': 'center', 'marginBottom': '40px'}))


    final_layout = html.Div(elements)
    
    return final_layout









########### APPLICATION CALLING ################################

if __name__ == '__main__':
    app.run(debug=True)