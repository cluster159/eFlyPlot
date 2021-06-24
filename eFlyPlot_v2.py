from neuprint import Client
import neuprint
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random as rd
import math
import pickle
import plotly
import dash_daq as daq
from pandas import DataFrame as Df
# import dash_get_synapse_v2 as dgs
from eFlyPlot_toolkit_v2 import *
import copy
import eFlyPlot_analysis as eFlyA
import json
from dash.dependencies import Input, Output
from collections import defaultdict
import eFlyPlot_layout

rd.seed(101)
'''
This script defines the basic callback functions and construct all the basic interface.
#notice: pymaid has similar purpose....
#notice: dash reusable is from https://github.com/plotly/dash-cytoscape/blob/master/usage-stylesheet.py


'''
class tmp_data_store:
    def __init__(self):
        self.button_history = {}
        self.data_history = {}




eflyplot_environment = eFlyPlot_environment()
app = eflyplot_environment.create_environment()
tmp_data = tmp_data_store()
# @app.callback(
#     Output('output-container-button','children'),
#     [Input("button", "n_clicks")],
#     [State]
# )
# def remove_history(n_clicks):
#
#     file_list=os.listdir(tmp_path)
#     for file in file_list:
#         os.remove(tmp_path+file)
#     return "The color history has been reset!"

########## network graph ########################
'''
edge的label是weight
app call back也要顯示neuronid 和 instance
classes是連結到的所有node id
'''

default_stylesheet = [
        {
            "selector": 'node',
            'style': {
                "opacity": 0.65,
            }
        },
        {
            "selector": 'edge',
            'style': {
                "curve-style": "bezier",
                "opacity": 0.65
            }
        },
    ]
@app.callback(Output('tap-node-json-output', 'children'),
              [Input('cytoscape', 'tapNode')])
def display_tap_node(data):
    return json.dumps(data, indent=2)


@app.callback(Output('tap-edge-json-output', 'children'),
              [Input('cytoscape', 'tapEdge')])
def display_tap_edge(data):
    return json.dumps(data, indent=2)


@app.callback(Output('cytoscape', 'layout'),
              [Input('dropdown-layout', 'value')])
def update_cytoscape_layout(layout):
    return {'name': layout}


@app.callback(Output('cytoscape', 'stylesheet'),
              [Input('cytoscape', 'tapNode'),
               Input('input-follower-color', 'value'),
               Input('input-following-color', 'value'),
               Input('dropdown-node-shape', 'value')])
def generate_stylesheet(node, follower_color, following_color, node_shape):
    if not node:
        return default_stylesheet

    stylesheet = [{
        "selector": 'node',
        'style': {
            'opacity': 0.3,
            'shape': node_shape
        }
    }, {
        'selector': 'edge',
        'style': {
            'opacity': 0.2,
            "curve-style": "bezier",
        }
    }, {
        "selector": 'node[id = "{}"]'.format(node['data']['id']),
        "style": {
            'background-color': '#B10DC9',
            "border-color": "purple",
            "border-width": 2,
            "border-opacity": 1,
            "opacity": 1,

            "label": "data(label)",
            "color": "#B10DC9",
            "text-opacity": 1,
            "font-size": 12,
            'z-index': 9999
        }
    }]

    for edge in node['edgesData']:
        if edge['source'] == node['data']['id']:
            stylesheet.append({
                "selector": 'node[id = "{}"]'.format(edge['target']),
                "style": {
                    'background-color': following_color,
                    'opacity': 0.9,
                    "label": "data(label)",
                }
            })
            stylesheet.append({
                "selector": 'edge[id= "{}"]'.format(edge['id']),
                "style": {
                    "mid-target-arrow-color": following_color,
                    "mid-target-arrow-shape": "vee",
                    "line-color": following_color,
                    'opacity': 0.9,
                    'z-index': 5000,
                    "label": "data(label)",

                }
            })

        if edge['target'] == node['data']['id']:
            stylesheet.append({
                "selector": 'node[id = "{}"]'.format(edge['source']),
                "style": {
                    'background-color': follower_color,
                    'opacity': 0.9,
                    'z-index': 9999,
                    "label": "data(label)",
                }
            })
            stylesheet.append({
                "selector": 'edge[id= "{}"]'.format(edge['id']),
                "style": {
                    "mid-target-arrow-color": follower_color,
                    "mid-target-arrow-shape": "vee",
                    "line-color": follower_color,
                    'opacity': 1,
                    'z-index': 5000,
                    "label": "data(label)",

                }
            })
    return stylesheet


#################################################
## get camera parameter
@app.callback(
    [Output('camera-text','value')],
    [Input("button_get_camera",'n_clicks')],
    [State("syn_graph", "figure"),State("history-store-synapse",'data')],
    prevent_initial_callbacks=True
)
def get_camera_position(b_get_camera,fig,record):
    try:
        camera_text = str(fig['layout']['scene']['camera'])
        print(fig['layout']['scene']['camera'])
        return [camera_text]
    except:
        return [" "]

## draw 3d plot
@app.callback(
    [Output("syn_graph", "figure"),Output('history-fig','data'),Output('s-up1_text','children'),Output('s-up2_text','children'),Output('s-up3_text','children')],
    [Input('s-up1','value'),Input('s-up2','value'),Input('s-up3','value'),Input('syn_button_draw', 'n_clicks'),Input('button_set_camera','n_clicks')],
    [State('history-store-synapse','data'),State("syn_graph", "figure"),State("up","value"),State("center","value"),State("eye","value"),State('history-fig','data'),State('camera-text','value'),State('s-up1_text','children'),State('s-up2_text','children'),State('s-up3_text','children')],
    prevent_initial_callbacks=True
)
def display_3d_plot(s1,s2,s3,b_draw,b_camera,history_data,fig,c_up,c_center,c_eye,c_data,camera_text,h_s1,h_s2,h_s3):
    fig=go.Figure(fig)
    camera = dict(up=dict(x=float(s1), y=float(s2), z=float(s3)))
    if (float(s1) != float(h_s1)) or (float(s2) != float(h_s2)) or (float(s3) != float(h_s3)):
        fig.update_layout(scene_camera=camera)
        print("jojo")
        return fig, c_data, s1, s2, s3
    print("nono")
    if b_camera:
        if b_camera > c_data['button_set_camera']:
            c_data['button_set_camera']=b_camera
            #########int or float????
            try:
                c_up = [float(i) for i in c_up.split(" ")]
            except:
                c_up = [0, 0, 1]
            if len(c_up)!=3:
                c_up = [0,0,1]

            try:
                c_center = [float(i) for i in c_center.split(" ")]
            except:
                c_center = [0, 0, 0]
            if len(c_center)!=3:
                c_center = [0,0,0]
            try:
                c_eye = [float(i) for i in c_eye.split(" ")]
            except:
                c_eye = [0,0,0]
            if len(c_eye)!=3:
                c_eye = [0,0,1]
            c_data['up'].append(c_up)
            c_data['center'].append(c_center)
            c_data['eye'].append(c_eye)
            camera = dict(
                up=dict(x=c_up[0], y=c_up[1], z=c_up[2]),
                center=dict(x=c_center[0], y=c_center[1], z=c_center[2]),
                eye=dict(x=c_eye[0], y=c_eye[1], z=c_eye[2])
            )
            fig = go.Figure(fig)
            fig.update_layout(scene_camera=camera)
            camera_text = f"up: {str(c_up)}\ncenter: {str(c_center)}\neye: {str(c_eye)}"
            return fig,c_data, s1, s2, s3
    ####deal with search
    print("DRAW")
    fig=go.Figure(layout=eflyplot_environment.get_layout_of_3dObject())
    for neuropil in history_data['NeuronSyn']['neuropil']:
        fig = eflyplot_environment.add_neuropil(neuropil,fig,history_data['NeuronSyn']['neuropil'][neuropil])
    print("draw", history_data['NeuronSyn']['neuron_name'])
    fig = eflyplot_environment.brain_space.get_neuron(history_data['NeuronSyn']['neuron_name'],fig)
    fig = eflyplot_environment.brain_space.get_xyz(history_data['NeuronSyn']['xyz_name'],fig)

    # connection = (upstream_id, up_w, downstream_id, down_w, syn_query_neuron, syn_roi,syn_rd)
    # color = set_color(select_color)
    # history_data['synapse'][connection] = color
    fig = eflyplot_environment.brain_space.neuprint_tool.get_synapse_pooled(history_data['NeuronSyn']['syn_search_query'],history_data['NeuronSyn']['upload-synapse_data'],fig)

    # for pair, select_color in zip(history_data['synapse'],history_data['synapse_color']):
    #     fig = get_synapse(pair[0],pair[1],fig,select_color)
    return fig,c_data, s1, s2, s3

def set_color(select_color): ##################需要在確認
    #################################################################################################
    try:
        original_select_color = select_color
        print(select_color)
        select_color = select_color[4:-1].split(",")
        print(f"processing{select_color}")
        r, g, b = int(select_color[0]), int(select_color[1]), int(select_color[2])
        # select_color = f'rgb({r},{g},{b},{a})'
        if r == 255 and g == 255 and b == 255:
            select_color = f'rgb({rd.randint(0, 255)},{rd.randint(0, 255)},{rd.randint(0, 255)})'
        else:
            select_color = original_select_color
    except:
        ####deal_with_color
        select_color = select_color['rgb']
        r, g, b, a = select_color['r'], select_color['g'], select_color['b'], select_color['a']
        select_color = f'rgb({r},{g},{b},{a})'
        if r == 255 and g == 255 and b == 255:
            select_color = f'rgb({rd.randint(0, 255)},{rd.randint(0, 255)},{rd.randint(0, 255)})'
        else:
            select_color = f'rgb({r},{g},{b},{a})'
    return select_color

## draw sankey graph of connection data

## draw nx graph of connection data
def generate_network_config(connection_data,node_type='bodyId',type_cluster=True):
    connection_data_list = copy.deepcopy(connection_data).values.tolist()
    print(connection_data_list)
    node_list = []
    id_edge_dict = {}
    type_edge_dict = defaultdict(list)
    id_to_type_dict = defaultdict(str)
    type_to_id_dict = defaultdict(list)
    node_connection_dict = defaultdict(list)
    for connection in connection_data_list:
        if connection[0] not in node_list:
            node_list.append(connection[0])
            if not connection[2]:
                connection[2] = 'unknown'
            id_to_type_dict[connection[0]] = connection[2]
            type_to_id_dict[connection[2]].append(connection[0])
        if connection[3] not in node_list:
            node_list.append(connection[3])
            if not connection[5]:
                connection[5] = 'unknown'
            id_to_type_dict[connection[3]] = connection[5]
            type_to_id_dict[connection[5]].append(connection[3])
        id_edge_dict[(connection[0],connection[3])] = connection[6]
        node_connection_dict[connection[0]].append(connection[3])
        node_connection_dict[connection[3]].append(connection[0])
        type_edge_dict[(connection[2],connection[5])].append(connection[6])
    cy_edges = []
    cy_nodes = []
    print("START")
    for node in node_connection_dict:
        classes = ""
        for node_adj in node_connection_dict[node]:
            classes = classes + str(node_adj) + " "
        classes = classes[:-1]
        cy_nodes.append({"data": {"id": f"{node}", "label": f"{id_to_type_dict[node]}_{node}"},"classes": classes})
    for network_edge in id_edge_dict:
        cy_edges.append({
            'data': {
                'source': f"{network_edge[0]}",
                'target': f"{network_edge[1]}",
                'label' : f"w: {id_edge_dict[network_edge]}",
                'weight': id_edge_dict[network_edge]
            },
            'classes':f"{network_edge[0]} {network_edge[1]}"
        })
    return cy_nodes,cy_edges

@app.callback(
    Output('cytoscape','elements'),
    [Input('conn_nx_button','n_clicks')],
    [State('history-store-connection','data'),State('cytoscape','elements')]
)
def generate_nx_graph(button_nx,history_data,elements):
    if button_nx:
        if button_nx > history_data['NeuronConn']['conn_nx_button']:
            history_data['NeuronConn']['conn_nx_button'] = button_nx
        if history_data['NeuronConn']['pooled_connection_data'] == 1:
            with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'rb')as ff:
                connection_data = pickle.load(ff)
        # print(connection_data)
        cy_nodes, cy_edges = generate_network_config(connection_data)
        return cy_edges+cy_nodes
    return elements

# def generate_nx_graph(button_nx,history_data):
#     if button_nx:
#         if button_nx > history_data['NeuronConn']['conn_nx_button']:
#             history_data['NeuronConn']['conn_nx_button'] = button_nx
#         if history_data['NeuronConn']['pooled_connection_data'] == 1:
#             with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'rb')as ff:
#                 connection_data = pickle.load(ff)
#             figure = eFlyA.get_network_graph(connection_data, node_type='bodyId', style='spring')
#             print("generated fig\n",figure)
#             return figure
#     figure = go.Figure()
#     return figure


## upload connection data, search connection data, clear connection data
@app.callback(
    [Output('conn_result_table','columns'),Output('conn_result_table','data'),Output('history-store-connection','data'),
     Output('upload-connection_data','filename'),Output('conn_connection_data','value'),Output("conn_query_neuron","value"),
     Output("conn_upstream_neuron",'value'),Output("conn_downstream_neuron",'value'),],
    [Input('conn_add_connection_button', 'n_clicks'),Input('conn_clear_button', 'n_clicks'),Input('conn_submit_button','n_clicks')],
    [State('upload-connection_data','filename'),State('conn_connection_data','value'),State("conn_query_neuron","value"),
     State('conn_brain_space','value'),State('conn_version','value'),State('conn_roi','value'),State("conn_upstream_neuron",'value'),
     State("conn_upstream_neuron_weight","value"),State("conn_downstream_neuron","value"),State("conn_downstream_neuron_weight","value"),
     State('history-store-connection','data')],
    prevent_initial_callbacks=True
)
def upload_conn_data(button_add, button_clear, button_submit, file_upload_list, file_input,query,template,version,roi,upn,upn_w,downn,down_w,history_data):
    print("Connection setup")
    columns_table = [{'name': 'up.bodyId', 'id': 'up.bodyId'}, {'name': 'up.type', 'id': 'up.type'},
               {'name': 'up.instance', 'id': 'up.instance'}, {'name': 'up.roiInfo', 'id': 'up.roiInfo'},{'name': 'down.bodyId', 'id': 'down.bodyId'}, {'name': 'down.type', 'id': 'down.type'},
               {'name': 'down.instance', 'id': 'down.instance'}, {'name': 'w.weight', 'id': 'w.weight'}, {'name': 'w.roiInfo', 'id': 'w.roiInfo'}]
    data_table = [{'up.bodyId': '', 'up.type': '', 'up.instance': '','down.bodyId': '', 'down.type': '', 'down.instance': '','w.weight':"", 'w.roiInfo': ''}]
    # if not upn and not downn and not roi and not query:
    #     return columns_table, data_table, history_data

    ## upload
    if button_add:
        print("get into upload process")
        if button_add > history_data['NeuronConn']['conn_add_connection_button']:
            history_data['NeuronConn']['conn_add_connection_button'] = button_add
            data = Df()
            file_list = []
            if file_upload_list:
                print("uploading", type(file_upload_list), file_upload_list)
                file_list = file_list + file_upload_list
            if file_input:
                groups = file_input.split(' ')
                for file_name in groups:
                    if not file_name:
                        continue
                    if file_name.find("Connection")==-1:
                        continue
                    if file_name.find("\n") != -1:
                        file_name = file_name[:-1]
                    if file_name.find("xlsx") == -1:
                        file_list.append(f"{file_name}.xlsx")
                    else:
                        file_list.append(file_name)

            if file_list:
                for file_id, file_name in enumerate(file_list):
                    if file_id == 0:
                        data = pd.read_excel(f"{eflyplot_environment.brain_space.connection_data_path}{file_name}")
                    else:
                        data = pd.merge(data, pd.read_excel(
                            f"{eflyplot_environment.brain_space.connection_data_path}{file_name}"), how='outer')
                data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


            if history_data['NeuronConn']['pooled_connection_data'] == 0 and len(data)!=0:
                history_data['NeuronConn']['pooled_connection_data'] = 1
                with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'wb')as ff:
                    pickle.dump(data,ff)
            else:
                if history_data['NeuronConn']['pooled_connection_data'] == 1:
                    with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'rb')as ff:
                        past_data = pickle.load(ff)
                    if len(data)==0:
                        data = past_data
                    else:
                        data = pd.merge(data, past_data,how='outer')
                if len(data)!=0:
                    history_data['NeuronConn']['pooled_connection_data'] = 1
                    with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'wb')as ff:
                        pickle.dump(data,ff)
            if len(data)!=0:
                result_table = copy.deepcopy(data)
                columns_table = [{"name": i, "id": i} for i in result_table.columns]
                data_table = copy.deepcopy(result_table).head().to_dict('records')
            print("upload over")
            return columns_table, data_table, history_data,[],"","","",""

    if button_clear:
        print("get into clear process")
        if button_clear > history_data['NeuronConn']['conn_add_connection_button']:
            history_data['NeuronConn']['conn_add_connection_button'] = button_clear
            history_data['NeuronConn']['pooled_connection_data'] = 0
            os.remove(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle")
            print("clear over")
            return columns_table, data_table, history_data,[],"","","",""

    if button_submit:
        print("get into submit process")
        if button_submit > history_data['NeuronConn']['conn_submit_button']:
            data = eflyplot_environment.brain_space.neuprint_tool.get_neuron_connection_all(query, roi, upn, upn_w,downn, down_w)
            if history_data['NeuronConn']['pooled_connection_data']==1 and len(data)!=0:
                with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'rb')as ff:
                    past_data = pickle.load(ff)
                data = pd.merge(data, past_data,how='outer')
                with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'wb')as ff:
                    pickle.dump(data,ff)
            elif len(data)!=0:
                history_data['NeuronConn']['pooled_connection_data'] = 1
                with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'wb')as ff:
                    pickle.dump(data,ff)
            elif history_data['NeuronConn']['pooled_connection_data'] == 1:
                with open(f"{eflyplot_environment.brain_space.tmp_path}conn_tmp.pickle", 'rb')as ff:
                    data = pickle.load(ff)
            if len(data)!=0:
                result_table = copy.deepcopy(data)
                columns_table = [{"name": i, "id": i} for i in result_table.columns]
                data_table = copy.deepcopy(result_table).to_dict('records')
            print("submit over")
            return columns_table, data_table, history_data,[],"","","",""
    print("Over")
    return columns_table,data_table, history_data,[],"","","",""


#self,query,roi,upn,upn_w,downn,down_w, result_info
## get neuron id info
@app.callback(
    [Output('info_result_table','columns'),Output('info_result_table','data')],
    [Input('info_button_submit','n_clicks')],
    [State("info_type","value"),State('info_brain_space','value'),State('info_version','value'),State('info_roi','value'),
     State("info_upstream_neuron",'value'),State("info_upstream_neuron_weight","value"),State("info_downstream_neuron","value"),
     State("info_downstream_neuron_weight","value"),State("info_result",'value')],
    prevent_initial_callbacks=True
)
def output_neuron_info(button_submit,query,template,version,roi,upn,upn_w,downn,down_w,result_info):
    if not upn and not downn and not roi and not query:
        columns = [{'name': 'n.bodyId', 'id': 'n.bodyId'}, {'name': 'n.type', 'id': 'n.type'}, {'name': 'n.instance', 'id': 'n.instance'}, {'name': 'n.roiInfo', 'id': 'n.roiInfo'}]
        data = [{'n.bodyId': '', 'n.type': '', 'n.instance': '', 'n.roiInfo': ''}]
        return columns, data
    print("Acquiring neuron info")
    print(query, template, version, roi, upn, upn_w, downn, down_w, result_info)
    result_table = []
    if template == "FlyEM":
        if eflyplot_environment.brain_space.neuprint_tool.version != version:
            eflyplot_environment.brain_space.neuprint_tool.connecting_neuprint_server()
        result_table = eflyplot_environment.brain_space.neuprint_tool.get_neuron_info_all(query,roi,upn,upn_w,downn,down_w,result_info)
        result_table = result_table
        columns=[{"name": i, "id": i} for i in result_table.columns]
        data=copy.deepcopy(result_table).to_dict('records')
        print(columns)
        print(data)
        return columns, data
    else:
        print("FC hasn't been constructed yet")
    if not result_table:
        return [{}], [{}]

# @app.callback(
#     [Output("syn_random","value")],
#     [Input("my-color-picker",'value')]
# )
# def change_color_mode(color):
#     if set_color(color) == f'rgb({255},{255},{255}':
#         return 'i'
#     return 's'

# ## set picture and get required info but not to draw
@app.callback(
    [Output('history-store-synapse',"data"),Output('my-color-picker', 'value'),Output('upload-neuron_data','filename'),
     Output('upload-xyz_data','filename'),Output('upload-synapse_data','filename')],
    [Input('syn_button_clear','n_clicks'),Input('syn_button_add_neuropil', 'n_clicks'),Input('syn_button_add_neuron', 'n_clicks'),
     Input('syn_button_add_xyz', 'n_clicks'),Input('syn_button_submit', 'n_clicks'), Input('syn_button_add_synapse','n_clicks')],
    [State("neuropil","value"),State("neuron_name","value"),State("xyz_name","value"),
     State('my-color-picker', 'value'),State("syn_roi",'value'), State("syn_query_neuron",'value'),
     State('syn_upstream_neuron','value'), State('syn_upstream_neuron_weight','value'),
     State('syn_downstream_neuron','value'), State('syn_downstream_neuron_weight','value'),
     State('history-store-synapse',"data"),
     State('upload-neuron_data','filename'),State('upload-xyz_data','filename'),
     State('syn_neuron_source_brain','value'),State('syn_neuron_version','value'),State('syn_neuron_label','value'),
     State('syn_xyz_source_brain','value'),State('syn_xyz_version','value'),State('syn_xyz_label','value'),
     State("syn_random",'value'),State('upload-synapse_data','filename')],
    prevent_initial_callbacks=True
)
def set_picture_variables(b_clear,b_neuropil,b_neuron,b_xyz,b_search, b_synapse,neuropil_names,neuron_names,xyz_names,
                          select_color,syn_roi,syn_query_neuron,upstream_id,up_w,downstream_id,down_w,history_data,
                          upload_neuron_list,upload_xyz_list,syn_neuron_source_brain,syn_neuron_version,syn_neuron_label,
                          syn_xyz_source_brain,syn_xyz_version,syn_xyz_label,syn_random,upload_synapse_list):
    print("1st",select_color)
    ####deal_with_clear
    if b_clear:
        if b_clear > history_data['NeuronSyn']['syn_button_clear']:
            history_data['NeuronSyn']['syn_button_clear'] = b_clear
            ## synapse

            if os.path.isfile(eflyplot_environment.brain_space.tmp_path + "search_button_click_record.pickle"):
                os.remove(eflyplot_environment.brain_space.tmp_path + "search_button_click_record.pickle")
            ## neuron
            if os.path.isfile(eflyplot_environment.brain_space.tmp_path + "syn_neuron.pickle"):
                os.remove(eflyplot_environment.brain_space.tmp_path + "syn_neuron.pickle")
            ## xyz
            if os.path.isfile(eflyplot_environment.brain_space.tmp_path + "syz_xyz.pickle"):
                os.remove(eflyplot_environment.brain_space.tmp_path + "syn_xyz.pickle")
            ## figure_all
            if os.path.isfile(eflyplot_environment.brain_space.tmp_path + "syz_3d.pickle"):
                os.remove(eflyplot_environment.brain_space.tmp_path + "syn_3d.pickle")
            return  eFlyPlot_layout.initialize_eFlyPlot_history(),dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}),[],[],[]
    ####deal_with_xyz
    if b_xyz:
        print("xyz",xyz_names)
        if b_xyz > history_data['NeuronSyn']['syn_button_add_xyz']:
            history_data['NeuronSyn']['syn_button_add_xyz']=b_xyz
            if xyz_names:
                xyz_names = xyz_names.split(" ")
            else:
                xyz_names = []
            if upload_xyz_list:
                xyz_names = xyz_names + upload_xyz_list
            final_xyz_names = []
            for xyz_name in xyz_names:
                if xyz_name.find(".txt")==-1:
                    final_xyz_names.append(f"{xyz_name}.txt")
                else:
                    final_xyz_names.append(f"{xyz_name}")
            print(f"setting {final_xyz_names}")
            if syn_random == 'u':
                color = set_color(select_color)
            for xyz_name in final_xyz_names:
                if not os.path.isfile(eflyplot_environment.brain_space.xyz_path+xyz_name):
                    print(f"<Warning> cannot open {xyz_name}")
                    continue
                if syn_random == 's':
                    select_color = set_color(select_color)
                elif syn_random == 'u':
                    select_color = color
                elif syn_random == 'i':
                    select_color =dict(rgb={'r': rd.randint(0,255), 'g': rd.randint(0,255), 'b': rd.randint(0,255), 'a': 1})
                history_data['NeuronSyn']['xyz_name'][xyz_name] = select_color
            return history_data,dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}),[],[],[]
    ####deal_with_neuron
    if b_neuron:
        if b_neuron > history_data['NeuronSyn']['syn_button_add_neuron']:
            print('neuron', neuron_names)
            history_data['NeuronSyn']['syn_button_add_neuron']=b_neuron
            if neuron_names:
                neuron_names=neuron_names.split(" ")
            else:
                neuron_names = []
            final_neuron_names = []
            if neuron_names:
                for neuron_name in neuron_names:
                    if neuron_name.find(".swc") == -1:
                        final_neuron_names.append(f"{neuron_name}_{syn_neuron_source_brain}_{syn_neuron_version}_{syn_neuron_label}.swc")
                    else:
                        final_neuron_names.append(f"{neuron_name}_{syn_neuron_source_brain}_{syn_neuron_version}_{syn_neuron_label}")
            if upload_neuron_list:
                final_neuron_names = final_neuron_names + upload_neuron_list
            print(f"setting {neuron_names}")
            if syn_random == "Uniform random color":
                select_color = dict(
                    rgb={'r': rd.randint(0, 255), 'g': rd.randint(0, 255), 'b': rd.randint(0, 255), 'a': 1})
                color = set_color(select_color)
            elif syn_random =="Selected color":
                color = set_color(select_color)
            for neuron_name in final_neuron_names:
                print("acquiring neuron..........")
                print(f'{eflyplot_environment.brain_space.skeleton_path}{neuron_name}')
                if not os.path.isfile(f'{eflyplot_environment.brain_space.skeleton_path}{neuron_name}'):
                    try:
                        c = Client('neuprint.janelia.org', dataset=f'hemibrain:{syn_neuron_version}', token=eflyplot_environment.brain_space.neuprint_tool.Token)
                        c.fetch_skeleton(copy.deepcopy(neuron_name).split("_")[0], export_path=f"{eflyplot_environment.brain_space.skeleton_path}{neuron_name}",format="swc")
                    except:
                        continue

                if syn_random =="Individual random color":
                    color =set_color(dict(rgb={'r': rd.randint(0,255), 'g': rd.randint(0,255), 'b': rd.randint(0,255), 'a': 1}))
                history_data['NeuronSyn']['neuron_name'][neuron_name] = color
                print(history_data['NeuronSyn']['neuron_name'][neuron_name])
            return history_data,dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}),[],[],[]

    ####deal_with_neuropil
    if b_neuropil is not None:
        if b_neuropil > history_data['NeuronSyn']['syn_button_add_neuropil']:
            history_data['NeuronSyn']['syn_button_add_neuropil'] = b_neuropil
            '''
            若有在history，就不更動，避免改到過去設定好的顏色
            但若是history有，現在沒有
            '''
            if syn_random == "Uniform random color":
                select_color = dict(
                    rgb={'r': rd.randint(0, 255), 'g': rd.randint(0, 255), 'b': rd.randint(0, 255), 'a': 1})
                color = set_color(select_color)
            elif syn_random == "Selected color":
                color = set_color(select_color)
            for neuropil in neuropil_names:
                if neuropil not in history_data['NeuronSyn']['neuropil'] :
                    if syn_random == "Individual random color":
                        color = set_color(dict(
                            rgb={'r': rd.randint(0, 255), 'g': rd.randint(0, 255), 'b': rd.randint(0, 255), 'a': 1}))
                    history_data['NeuronSyn']['neuropil'][neuropil] = color

                # elif neuropil in history_data['neuropil'] andhistory_data['neuropil'].index(neuropil)
            for neuropil in history_data['NeuronSyn']['neuropil']:
                if neuropil not in neuropil_names:
                    del(history_data['NeuronSyn']['neuropil'][neuropil])
            return history_data, dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}),[],[],[]

    ####deal with search
    if b_search:
        if b_search > history_data['NeuronSyn']['syn_button_submit']:
            history_data['NeuronSyn']['syn_button_submit'] = b_search
            if not upstream_id:
                upstream_id = "noinfo"
            if not up_w:
                up_w = 0
            if not downstream_id:
                downstream_id = "noinfo"
            if not down_w:
                down_w = 0
            if not syn_query_neuron:
                syn_query_neuron = "noinfo"
            if not syn_roi:
                syn_roi = "noinfo"
            connection = f"{upstream_id} {up_w} {downstream_id} {down_w} {syn_query_neuron} {syn_roi}"
            color = set_color(select_color)
            history_data['NeuronSyn']['syn_search_query'][connection] = color
            return history_data, dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}),[],[],[]

    if b_synapse:
        if b_synapse > history_data['NeuronSyn']['syn_button_add_synapse']:
            history_data['NeuronSyn']['syn_button_add_synapse'] = b_synapse
            if upload_synapse_list:
                final_synapses = upload_synapse_list
            if syn_random == "Uniform random color":
                select_color = dict(
                    rgb={'r': rd.randint(0, 255), 'g': rd.randint(0, 255), 'b': rd.randint(0, 255), 'a': 1})
                color = set_color(select_color)
            elif syn_random == "Selected color":
                color = set_color(select_color)
            for synapse_file in final_synapses:
                if syn_random == "Individual random color":
                    color = set_color(dict(
                        rgb={'r': rd.randint(0, 255), 'g': rd.randint(0, 255), 'b': rd.randint(0, 255), 'a': 1}))
                history_data['NeuronSyn']['upload-synapse_data'][synapse_file] = color
        return history_data, dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}), [], [], []
    return history_data, dict(rgb={'r': 255, 'g': 255, 'b': 255, 'a': 1}),[],[],[]


if __name__=='__main__':
    app.run_server(debug=True)





