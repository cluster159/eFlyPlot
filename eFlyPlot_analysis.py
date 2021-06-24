import networkx as nx
import plotly.graph_objects as go
import pandas
from pandas import DataFrame
'''
Function plan
1. compare connection number to same subtype in same neuropil
2. kde map of points
3. compare spatial distribution similarity in a given neuropil
4. analyze neuronal path length etc.
5. connection map (networkx etc.)
6. sankey chart for info flow (acyclic)
7. clustering
8. read simulation results for further visualization

'''
'''
 {
        'data': {'id': 'one', 'label': 'Modified Color'},
        'position': {'x': 75, 'y': 75},
        'classes': 'red' # Single class
    },
'''
import copy
from collections import defaultdict


def generate_network_file(connection_data,node_type='bodyId',type_cluster=True):
    connection_data_list = copy.deepcopy(connection_data).values.tolist()
    node_list = []
    id_edge_dict = {}
    type_edge_dict = defaultdict(list)
    id_to_type_dict = defaultdict(str)
    type_to_id_dict = defaultdict(list)
    f = open('network_sample.txt','wt')
    for connection in connection_data_list:
        if connection[1] not in node_list:
            node_list.append(connection[1])
            if not connection[3]:
                connection[3] = 'unknown'
            id_to_type_dict[connection[1]] = connection[3]
            type_to_id_dict[connection[3]].append(connection[1])
        if connection[4] not in node_list:
            node_list.append(connection[4])
            if not connection[6]:
                connection[6] = 'unknown'
            id_to_type_dict[connection[4]] = connection[6]
            type_to_id_dict[connection[6]].append(connection[4])
        id_edge_dict[(connection[1],connection[4])] = connection[7]
        type_edge_dict[(connection[3],connection[4])].append(connection[7])
        f.writelines(f"{connection[1]} {connection[4]}\n")
    f.close()

def generate_cytoscope(connection_data,node_type='bodyId',type_cluster=True):
    connection_data_list = copy.deepcopy(connection_data).values.tolist()
    node_list = []
    id_edge_dict = {}
    type_edge_dict = defaultdict(list)
    id_to_type_dict = defaultdict(str)
    type_to_id_dict = defaultdict(list)
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
        type_edge_dict[(connection[2],connection[3])].append(connection[6])


    for node in node_list:
        data = {'id': f'{node}', 'label': f'{id_to_type_dict[node]}'}
        element = {}
        element['data'] = data
        if type_cluster:
            element['position']


def get_network_graph(connection_data,node_type='bodyId',style='spring'):
    Graph = nx.from_pandas_edgelist(connection_data, source=f'up.{node_type}', target=f'down.{node_type}', edge_attr=True, create_using=nx.DiGraph())
    if style == 'spring':
        pos = nx.spring_layout(Graph)
    figure = networkGraph(Graph,pos)
    return figure

def networkGraph(G,pos):

    # edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False,
        mode='lines')

    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=text,
        mode='markers+text',
        showlegend=False,
        hoverinfo='none',
        marker=dict(
            color='pink',
            size=50,
            line=dict(color='black', width=1)))

    # layout
    layout = dict(plot_bgcolor='white',
                  paper_bgcolor='white',
                  margin=dict(t=10, b=10, l=10, r=10, pad=0),
                  xaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True),
                  yaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True))

    # figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig

if __name__=='__main__':
    connection_data = pandas.read_excel('Output/Connection_data/Connection_sKCs_downstream_of_sDA1s_sDM1s_w_0_v1.2.1.xlsx')
    generate_network_file(connection_data)
