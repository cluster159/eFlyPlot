import dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from pandas import DataFrame as Df
import dash_daq as daq
import plotly.graph_objects as go
from dash.dependencies import Input, Output,State
import dash_cytoscape as cyto
import json
import dash_reusable_components as drc
import test_cyto
import dash_extensions as des
def get_neuropil_layout(neuropil_options=[{'label':'CA(R)','value':'CA(R)'}],template='FlyEM'):
    neuropil_options = [{'label':f"{neuropil}",'value':f"{neuropil}"} for neuropil in neuropil_options]
    neuropil_layout = html.Div([
                        html.P(f"{template}"),
                        html.P("Choose neuropils:"),
                        dcc.Checklist(
                            id="neuropil",options=neuropil_options,value=[],labelStyle={'display': 'inline-block'}),
                        html.Button('Add neuropil', id='syn_button_add_neuropil'),
    ])
    return neuropil_layout

def get_brain_space_options():
    options = [
        {"label": "FlyEM", "value": "FlyEM"},
        {"label": "FlyCircuit", "value": "FlyCircuit"},

    ]
    return options

def get_neuron_label_options(source="FlyEM"):
    if source == "FlyEM":
        options = [
            {"label": "Original", "value": "Original"},
            {"label": "CCC", "value": "CCC"},
        ]
    elif source == "FlyCircuit":
        options = [
            {"label": "v1.2", "value": "v1.2"},
            {"label": "v2.0", "value": "v2.0"},
        ]
    return options

def get_version_options(template="FlyEM"):
    if template=="FlyEM":
        options = [
            {"label": "v0.9", "value": "v0.9"},
            {"label": "v1.01", "value": "v1.01"},
            {"label": "v1.1", "value": "v1.1"},
            {"label": "v1.2.1", "value": "v1.2.1"}
        ]
    elif template=="FlyCircuit":
        options = [
            {"label": "v1.2", "value": "v1.2"},
            {"label": "v2.0", "value": "v2.0"},
        ]
    else:
        options = []
    return options

def initialize_eFlyPlot_fig_history():
    history_fig = {'up': [], 'center': [], 'eye': [], 'history-fig': {}, 'button_set_camera': 0, 'button_get_camera': 0,
                   'syn_button_draw': 0}

    return history_fig

def initialize_eFlyPlot_history():
    history = {'NeuronInfo': {'info_brain_space': 'FlyEM', 'info_version': "v1.2.1", "info_type": [], "info_roi": [],
                    "info_upstream_neuron": [],
                    "info_upstream_neuron_weight": 0, "info_downstream_neuron": [], "info_downstream_neuron_weight": 0,
                    'info_button_submit': 0,'info_button_download':0,'info_file_name':[],
                    "info_result": ['type', 'id', 'roi'], 'info_result_table': []},
     'NeuronConn': {'conn_brain_space': 'FlyEM', 'conn_version': 'v1.2.1', 'conn_roi': [], "conn_upstream_neuron": [],
                    "conn_upstream_neuron_weight": 0, 'conn_query_neuron': [], "conn_downstream_neuron": [],
                    "conn_downstream_neuron_weight": 0, 'conn_submit_button': 0, 'conn_clear_button': 0,
                    "conn_connection_data": [], 'upload-connection_data': [], 'conn_add_connection_button': 0,
                    'pooled_connection_data': 0,'conn_button_download':0,'conn_file_name':[],
                    'conn_nx_button': 0, 'conn_sk_button': 0, 'conn_nx_graph': go.Figure(),
                    'conn_sk_graph': go.Figure()},
     'NeuronSyn': {'syn_brain_space': "FlyEM", "syn_version": 'v1.2.1', "neuron_name": {},
                   "syn_neuron_source_brain": "FlyEM",
                   "syn_neuron_version": "v1.2.1", "syn_neuron_label": "Original", 'upload-neuron_data': [],
                   'syn_button_add_neuron': 0, "xyz_name": {}, "syn_xyz_source_brain": "FlyEM", "syn_xyz_version": "",
                   "syn_xyz_label": "", 'upload-xyz_data': [], 'syn_button_add_xyz': 0, 'syn_roi': [],
                   'syn_upstream_neuron': [],'syn_upstream_neuron_weight': 0, "syn_query_neuron": [], 'syn_downstream_neuron': [],
                   'syn_downstream_neuron_weight': 0, 'syn_button_submit': 0, "neuropil": {},
                   'my-color-picker': dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}), 'syn_button_reset': 0,
                    'syn_button_clear': 0, "syn_graph": go.Figure(),'syn_button_download':0,'syn_file_name':[],
                   'camera-text': "", 'syn_button_add_neuropil': 0,'upload-synapse_data': {}, 'syn_button_add_synapse': 0,
                   'syn_search_query':{}
                   }
     }
    return history

def get_tab_for_eFlyPlot_info(label="eFlyPlot"):
    tab = dcc.Tab(label=label, children=[
            dcc.Markdown('''
                #### Brief Introduction to eFlyPlot 
                This program is developed by Ching-Che Charng @ National Tsing-Hua University. Currently, it is a beta version.\n
                If you have any problem, please contact [C.C. Charng](cluster159@gmail.com)\n
                To get original FlyEM data, please visit https://neuprint.janelia.org/ The EM data is published by the research team at Janelia Research Campus, USA.\n
                To get original FlyCircuit data, please visit http://www.flycircuit.tw/ The Flyroscent data is published by the team at Brain Research Center, Taiwan\n
                The detail instruction of neuprint package is at https://neuprint.janelia.org/public/neuprintuserguide.pdf\n
                \n
                Currently, we support four kinds of visualization for FlyEM dataset:\n
                (i) neuropil
                (ii) neuron_ids (seperated by space)
                (iii) xyz file
                (iv) synapse of a neuron or a group of neurons
                '''),
            dcc.Store(id='history-store',data= initialize_eFlyPlot_history()),
            dcc.Store(id='history-store-connection', data=initialize_eFlyPlot_history()),
            dcc.Store(id='history-store-synapse', data=initialize_eFlyPlot_history()),
            dcc.Store(id='history-store-dendro',data=initialize_eFlyPlot_history()),

        dcc.Store(id='history-fig', data=initialize_eFlyPlot_fig_history())
    ])


    return tab

def get_tab_for_basic_info(label="NeuronInfo"):
    tab = dcc.Tab(label=label, children=[
        html.Div(
            [html.Div([
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='info_brain_space',
                            options=get_brain_space_options(),
                            value='FlyEM'
                        ),
                    ], style={'display': 'inline-block',"height": "auto", "width":"20%"}),
                    html.Div([
                        dcc.Dropdown(
                            id='info_version',
                            options=get_version_options(),
                            value='v1.2.1'
                        ),
                    ],style={'display':'inline-block',"height": "auto", "width":"20%"}),
                ]),
                html.P("Search Condition", style={'fontsize': 30}),
                html.Div([html.P('Query neuron type or neuron id', style={"height": "auto", "margin-bottom": "auto"}),
                          dcc.Input(id="info_type", type="text", placeholder="input type {}".format("text"),
                                    debounce=True, ), ]),
                html.Div([html.P('ROI', style={"height": "auto", "margin-bottom": "auto"}),
                          dcc.Input(id="info_roi", type="text",
                                    placeholder="input type {}".format("text"),
                                    debounce=True, ), ]),
                html.P("[WARNING]: If you are setting connection condition, you have to make sure the bodyId should in return data."),
                html.Div([
                    html.Div([html.P("Upstream neuron", style={"height": "auto", "margin-bottom": "auto"}),
                              dcc.Input(id="info_upstream_neuron", type="text",
                                        placeholder="input type {}".format("text"),
                                        debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([html.P("Weight threshold",
                                     style={"height": "auto", "margin-bottom": "auto"}),
                              dcc.Input(id="info_upstream_neuron_weight", type="number",
                                        placeholder="input type {}".format("number"),
                                        debounce=True, ), ], style={'display': 'inline-block'})]),
                html.Div([
                    html.Div([html.P("Downstream neuron", style={"height": "auto", "margin-bottom": "auto"}),
                              dcc.Input(id="info_downstream_neuron", type="text",
                                        placeholder="input type {}".format("text"),
                                        debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([html.P("Weight threshold",
                                     style={"height": "auto", "margin-bottom": "auto"}),
                              dcc.Input(id="info_downstream_neuron_weight", type="number",
                                        placeholder="input type {}".format("number"),
                                        debounce=True, ), ], style={'display': 'inline-block'}),
                ]),
                html.Div([
                    html.P("Return Parameters", style={'fontsize': 30}),
                    dcc.Checklist(
                        id="info_result",
                        options=[{'label': 'neuron_type', 'value': 'type'}, {'label': 'neuron_id', 'value': 'id'},
                                 {'label': 'roi', 'value': 'roi'}],
                        value=['type', 'id', 'roi'],
                        labelStyle={'display': 'inline-block'}
                    )
                ]),
                html.Div([html.Button('Submit', style={"height": "auto", "margin-bottom": "auto"},
                                      id='info_button_submit'), ]),
                dcc.Store(id='info_name_list',data={'options':[]}),
                dcc.Dropdown(
                    id='info_download_dropdown',
                    options=[],
                    multi=True
                ),
                html.Button("Download", id="info_button_download"),
                dcc.Download(id="info_download_xlsx"),
                dash_table.DataTable(
                    id='info_result_table',
                    style_cell={'textAlign': 'left'},
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'Region'},
                            'textAlign': 'left'
                        }
                    ]
                    #     columns=[],
                    #     data=Df(data=[])),
                ),


            ])
            ])])
    return tab

def get_tab_for_neuron_connection(label="NeuronConn"):
    tab = dcc.Tab(label=label, children=[
        html.Div(
            [html.Div([
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='conn_brain_space',
                            options=get_brain_space_options(),
                            value='FlyEM'
                        ),
                    ], style={'display': 'inline-block', "height": "auto", "width": "20%"}),
                    html.Div([
                        dcc.Dropdown(
                            id='conn_version',
                            options=get_version_options(),
                            value='v1.2.1'
                        ),
                    ], style={'display': 'inline-block', "height": "auto", "width": "20%"}),
                ]),
                html.P("Type the connection file name or select connection files"),
                dcc.Input(
                    id="conn_connection_data",
                    type="text",
                    placeholder="input type {}".format("text"),
                    debounce=True,
                    value=""
                ),
                dcc.Upload(
                    id='upload-connection_data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Connection Data Files (.xlsx)')
                    ]),
                    style={
                        'width': '30%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Button('Add connection files', id='conn_add_connection_button'),
                html.P("Obtain connection results", style={'fontsize': 30}),
                html.P("Search Condition", style={'fontsize': 30}),
                html.Div([
                    html.Div([html.P("Upstream neuron", style={"height": "auto", "margin-bottom": "auto"}),
                              dcc.Input(id="conn_upstream_neuron", type="text",
                                        placeholder="input type {}".format("text"),
                                        debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([html.P("Weight threshold",
                                     style={"height": "auto", "margin-bottom": "auto"}),
                              dcc.Input(id="conn_upstream_neuron_weight", type="number",
                                        placeholder="input type {}".format("number"),
                                        debounce=True, ), ], style={'display': 'inline-block'}),
                    html.Div([
                        html.P("Query neuron", style={"height": "auto", "margin-bottom": "auto"}),
                        dcc.Input(id="conn_query_neuron", type="text",
                                  placeholder="input type {}".format("text"),
                                  debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([
                        html.P("Downstream neuron", style={"height": "auto", "margin-bottom": "auto"}),
                        dcc.Input(id="conn_downstream_neuron", type="text",
                                  placeholder="input type {}".format("text"),
                                  debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([
                        html.P("Weight threshold",
                               style={"height": "auto", "margin-bottom": "auto"}),
                        dcc.Input(id="conn_downstream_neuron_weight", type="number",
                                  placeholder="input type {}".format("number"),
                                  debounce=True, ), ], style={'display': 'inline-block'}),

                ]),
                html.Div([html.P('ROI', style={"height": "auto", "margin-bottom": "auto"}),
                          dcc.Input(id="conn_roi", type="text",
                                    placeholder="input type {}".format("text"),
                                    debounce=True, ), ]),
                html.Div([html.Button('Submit', style={"height": "auto", "margin-bottom": "auto"},
                                      id='conn_submit_button'), ]),
                dcc.Store(id='conn_name_list',data={'options':[]}),
                dcc.Dropdown(
                    id='conn_download_dropdown',
                    options=[],
                    multi=True
                ),

                html.Button("Download Excel", id="conn_button_download"),
                dcc.Download(id="conn_download_xlsx"),
                html.Div([html.Button('Clear connection data', style={"height": "auto", "margin-bottom": "auto"},
                                      id='conn_clear_button'), ]),
                html.P(
                    "Note: To generate analysis chart, please submit the query job first. It will pool the connection data together!!"),
                html.Div(
                    [html.Button('Generate network configuration', style={"height": "auto", "margin-bottom": "auto"},
                                 id='conn_nx_button'), ]),
                html.Div(
                    [html.Button('Generate sankey chart (acyclic)', style={"height": "auto", "margin-bottom": "auto"},
                                 id='conn_sk_button'), ]),
                dash_table.DataTable(
                    id='conn_result_table',
                    style_cell={'textAlign': 'left'},
                    style_cell_conditional=[
                        {
                            'if': {'column_id': 'Region'},
                            'textAlign': 'left'
                        }
                    ]
                    #     columns=[],
                    #     data=Df(data=[])),
                ),
            ]),
                html.P("Please specify the source neuron instance or id and the target neuron"),
                html.Div([
                    html.P("Source neuron", style={"height": "auto", "margin-bottom": "auto"}),
                    dcc.Input(id="conn_source_neuron", type="text",
                              placeholder="input type {}".format("text"),
                              debounce=True, )], style={'display': 'inline-block'}),
                html.Div([
                    html.P("Target neuron",style={"height": "auto", "margin-bottom": "auto"}),
                    dcc.Input(id="conn_target_neuron", type="text",
                              placeholder="input type {}".format("text"),
                              debounce=True, ), ], style={'display': 'inline-block'}),
                dcc.Graph(id="conn_sk_graph", figure=go.Figure()),
            ])])
    return tab

def get_tab_for_neuron_synapses(label="NeuronSyn",neuropil_options=[{'label':'CA(R)','value':'CA(R)'}]):
    tab = dcc.Tab(label=label,children=[
        html.Div(
        [   html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='syn_brain_space',
                            options=get_brain_space_options(),
                            value='FlyEM'
                        ),
                    ], style={'display': 'inline-block',"height": "auto", "width":"20%"}),
                    html.Div([
                        dcc.Dropdown(
                            id='syn_version',
                            options=get_version_options(),
                            value='v1.2.1'
                        ),
                    ],style={'display':'inline-block',"height": "auto", "width":"20%"}),
            ]),
            html.Div([
            html.P(
                "Neuron types or ids"),
            dcc.Input(
                id="neuron_name",
                type="text",
                placeholder="input type {}".format("text"),
                debounce=True,
                value=""
            ),
            html.P("Source brain"),
            dcc.Dropdown(
                id='syn_neuron_source_brain',
                options=get_brain_space_options(),
                value='FlyEM'
            ),
            dcc.Dropdown(
                id='syn_neuron_version',
                options=get_version_options("FlyEM"),
                value='v1.2.1'
            ),
            dcc.Dropdown(
                id='syn_neuron_label',
                options=get_neuron_label_options(),
                value='Original'
            ),

            dcc.Upload(
                id='upload-neuron_data',
                children=html.Div([
                    'Drag and drop or select Neuron Files'
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Button('Add neuron', id='syn_button_add_neuron'),

        ], style={'display':'inline-block'}),
        html.Div([
            html.P(
                "Other xyz file with path(.txt)"),
            dcc.Input(
                id="xyz_name",
                type="text",
                placeholder="input type {}".format("text"),
                debounce=True,
                value=""
            ),
            html.P("Source brain"),
            dcc.Dropdown(
                id='syn_xyz_source_brain',
                options=get_brain_space_options(),
                value='FlyEM'
            ),
            dcc.Dropdown(
                id='syn_xyz_version',
            ),
            dcc.Dropdown(
                id='syn_xyz_label',
            ),
            dcc.Upload(
                id='upload-xyz_data',
                children=html.Div([
                    'Drag and drop or select xyz files'
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Button('Add xyz file', id='syn_button_add_xyz'),
        ], style={'display':'inline-block'}),
            html.Div([
                html.P("Search Condition", style={'fontsize':30}),
                html.Div([html.P('ROI', style={"height": "auto", "margin-bottom": "auto"}),
                          dcc.Input(id="syn_roi", type="text",
                                    placeholder="input type {}".format("text"),
                                    debounce=True, ), ]),
                html.Div([
                    html.Div([html.P("Upstream neuron", style={"height": "auto", "margin-bottom": "auto"}),
                          dcc.Input(id="syn_upstream_neuron", type="text",
                                    placeholder="input type {}".format("text"),
                                    debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([html.P("Weight threshold",
                                 style={"height": "auto", "margin-bottom": "auto"}),
                          dcc.Input(id="syn_upstream_neuron_weight", type="number",
                                    placeholder="input type {}".format("number"),
                                    debounce=True, ), ], style={'display': 'inline-block'}),
                    html.Div([
                        html.P("Query neuron", style={"height": "auto", "margin-bottom": "auto"}),
                        dcc.Input(id="syn_query_neuron", type="text",
                                  placeholder="input type {}".format("text"),
                                  debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([
                        html.P("Downstream neuron", style={"height": "auto", "margin-bottom": "auto"}),
                        dcc.Input(id="syn_downstream_neuron", type="text",
                            placeholder="input type {}".format("text"),
                            debounce=True, )], style={'display': 'inline-block'}),
                    html.Div([
                        html.P("Weight threshold",
                            style={"height": "auto", "margin-bottom": "auto"}),
                        dcc.Input(id="syn_downstream_neuron_weight", type="number",
                            placeholder="input type {}".format("number"),
                            debounce=True, ), ], style={'display': 'inline-block'}),

                ]),
            html.Div([html.Button('Submit', style={"height": "auto", "margin-bottom": "auto"},
                                  id='syn_button_submit'), ]),
            dcc.Store(id='syn_name_list',data={'options':[]}),
            dcc.Dropdown(
                    id='syn_download_dropdown',
                    options=[],
                    multi=True
            ),
            html.Button("Download Excel", id="syn_button_download"),
            dcc.Download(id="syn_download_xlsx"),
            dcc.Upload(
                id='upload-synapse_data',
                children=html.Div([
                        'Drag and drop or select Synapse Files(.xlsx)'
                    ]),
                style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Button('Add synapse data', id='syn_button_add_synapse'),

                get_neuropil_layout(neuropil_options),


                ###### ColorPicker ###############
            html.Div([dcc.Dropdown(id='syn_random',options=[{"label":"Uniform random color","value":"Uniform random color"},
                                                            {"label":"Individual random color","value":"Individual random color"},
                                                            {"label":"Selected color","value":"Selected color"}],value="Selected color"),
            ],style={"width":"30%"}),

            daq.ColorPicker(
                    id='my-color-picker',
                    label='Select another neuropil color',
                    value=dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1})
            ),
            html.Button('Reset_color', id='syn_button_reset'),
            html.Button('Start to draw', id='syn_button_draw'),
            html.Button('Clear', id='syn_button_clear'),
            html.Div(id='color-picker-output'),
            dcc.Graph(id="syn_graph", figure=go.Figure()),
            html.P("Set camera"),
            dcc.Input(
                id="up",
                type="text",
                placeholder="input type {}".format("text"),
                debounce=True,
                value=""
            ),
                dcc.Input(
                    id="center",
                    type="text",
                    placeholder="input type {}".format("text"),
                    debounce=True,
                    value=""
                ),
                dcc.Input(
                    id="eye",
                    type="text",
                    placeholder="input type {}".format("text"),
                    debounce=True,
                    value=""
                ),
                daq.Slider(
                    min=0, max=360, value=0, id='s-up1', step=1, size=360,
                    marks={'60': '60', '120': '120', '180': '180', '240': '240', '300': '300', '360': '360'}
                ),
                html.Div(id='s-up1_text', children='0'),
                daq.Slider(
                    min=0, max=360, value=0, id='s-up2', step=1, size=360,
                    marks={'60': '60', '120': '120', '180': '180', '240': '240', '300': '300', '360': '360'}
                ),
                html.Div(id='s-up2_text', children='0'),
                daq.Slider(
                    min=0, max=360, value=1, id='s-up3', step=1, size=360,
                    marks={'60': '60', '120': '120', '180': '180', '240': '240', '300': '300', '360': '360'}
                ),
                html.Div(id='s-up3_text', children='1'),
                html.Button('set camera', id='button_set_camera'),
                html.Button('get camera', id='button_get_camera'),
                dcc.Textarea(
                    id='camera-text',
                    value='',
                    # style={'width': '100%', 'height': 300},
                ),


            ])

    ])])
    return tab

def get_dendrogram():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=random_x, y=random_y0,
                             mode='lines',
                             name='lines'))
    fig.add_trace(go.Scatter(x=random_x, y=random_y1,
                             mode='lines+markers',
                             name='lines+markers'))
    fig.add_trace(go.Scatter(x=random_x, y=random_y2,
                             mode='markers', name='markers'))

    fig.show()

def get_tab_for_synapse_analysis():
    tab = dcc.Tab(label="NeuSynAnalysis", children=[
        dcc.Tabs(children=[
            dcc.Tab(children=[
            html.P("After obtaining synapses in NeuronSyn, the synapse distribution can be visualize here."),
            html.P("Please specify which brain region you want to analyze."),
            dcc.Dropdown(id='spatial_roi',
                         options=[],
                         ),
            html.Img(id='Spatial_distribution')

        ]),
        dcc.Tab(label="SynDendrogram", children=[
            html.P("Visualization of synaptic arrangement of single neuron!"),
            html.P("<WARNING> "
                   "1. Due to skeleton tracing issues, some neurons or part of neuron branches cannot present here.\n "
                   "2. Currently, the synapse distance threshold is 3 um. If the synapse position is farer than threshold, the synapse will be discared.\n"
                   ),
            html.P("Target neuronId."),
            dcc.Input(id="syn_dendrogram_neuronId", type="text", placeholder="input type {}".format("text"),
                      debounce=True, ),
            html.P("Target neuropil."),
            dcc.Input(id="syn_dendrogram_synapse_neuropil", type="text", placeholder="input type {}".format("text"),
                      debounce=True, ),
            html.P("Synaptic cmaps for following files in order."),
            dcc.Input(id="syn_dendrogram_synapse_cmap", type="text", placeholder="input type {}".format("text"),
                      debounce=True, ),
            html.Button('Draw syn-dendrogram', id='button_dendro'),
            dcc.Dropdown(
                id='dendro_download_dropdown',
                options=[],
                multi=True
            ),
            # dcc.Graph(id='syn_dendrogram', figure=go.Figure())
            html.Img(id='syn_dendrogram')

        ]),
        dcc.Tab(label="Brain registration", children=[
            html.P("Currently the brain registration function is offered by navis.\nPlease visit the following website to get more information: https://github.com/schlegelp/navis-flybrains\n"),
            # dcc.Input(id="warping_xyz", type="text", placeholder="input type {}".format("text"),
            #           debounce=True, ),
            dcc.Upload(
                id='upload-warping_data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select xyz Data Files (.xlsx)')
                ]),
                style={
                    'width': '30%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            dcc.Download(id="warp_download"),
            html.Div([
                html.P("Source brain"),
                dcc.Dropdown(
                    id='warping_source',
                    options=[{'label': "FlyEM", 'value': 'FlyEM'},
                             {'label': "FlyCircuit", 'value': 'FlyCircuit'},
                             {'label': "FlyEM_um", 'value': 'FlyEM_um'},
                             {'label': "FCWB", 'value': 'FCWB'},
                             {'label': "JRC2018U", 'value': 'JRC2018U'},
                             {'label': "JRC2018F", 'value': 'JRC2018F'},
                             {'label': "JRC2018Fraw", 'value': 'JRC2018Fraw'},
                             {'label': "vfb", 'value': 'vfb'},
                             {'label': "FAFB", 'value': 'FAFB'}],
                    value='FlyEM'
                ),
            ]),
            html.Div([
                html.P("Target brain"),
                dcc.Dropdown(
                    id='warping_target',
                    options=[{'label': "FlyEM", 'value': 'FlyEM'},
                             {'label': "FlyCircuit", 'value': 'FlyCircuit'},
                             {'label': "FlyEM_um", 'value': 'FlyEM_um'},
                             {'label': "FCWB", 'value': 'FCWB'},
                             {'label': "JRC2018U", 'value': 'JRC2018U'},
                             {'label': "JRC2018F", 'value': 'JRC2018F'},
                             {'label': "JRC2018Fraw", 'value': 'JRC2018Fraw'},
                             {'label': "vfb", 'value': 'vfb'},
                             {'label': "FAFB", 'value': 'FAFB'}],
                    value='FlyCircuit'
                ),
            ]),
            html.Button('Start warping coordinates', id='button_warping'),
        ]),
                 ])
    ])
    return tab


def get_tab_for_neuron_connection_analysis():
    # tab = dcc.Tab(label="NeuConnAnalysis", children=[
    #     # html.Div([
    #     # dcc.Tab(label="Cytoscape", children=[
    #     test_cyto.get_tab_cytoscape_layout(),
    #     # ]),
    #     # dcc.Tab(label="Correlation_matrix", children=[
    #     #     html.P("Please specify your target neurons and we will analyze the connection correlations using connection data in your result which connects to your target neurons."),
    #     #     dcc.Input(id="conn_ana_source", type="text", placeholder="input type {}".format("text"),
    #     #               debounce=True, ), ]),
    #     #     dcc.Dropdown(
    #     #         id='conn_ana_correlation_type',
    #     #         options=[{'label':"Pearson correlation",'value':'P'},{'label':"Cosine correlation",'value':'C'},{'label':"Jaccard correlation",'value':'J'}],
    #     #         value='P'
    #     #     ),
    #     #     ])
    #
    # ])
    tab =  test_cyto.get_tab_cytoscape_layout()
    return tab

if __name__ == '__main__':
    app = dash.Dash(__name__, prevent_initial_callbacks=True)
    app.layout = html.Div(id='stuff', children=[
        dcc.Tabs([
            get_tab_for_eFlyPlot_info(),
            get_tab_for_basic_info(),
            get_tab_for_neuron_connection(),
            get_tab_for_neuron_connection_analysis(),
            get_tab_for_neuron_synapses(),
            get_tab_for_synapse_analysis()
        ])
    ])
    app.run_server(debug=True)