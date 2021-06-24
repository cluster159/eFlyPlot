import json

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import dash_cytoscape as cyto
import dash_reusable_components as drc

# ###################### DATA PREPROCESSING ######################
def process_data():
    # Load data
    with open('network_sample.txt', 'r') as f:
        network_data = f.read().split('\n')
    print(network_data)
    # We select the first 750 edges and associated nodes for an easier visualization
    edges = network_data[:750]
    nodes = set()

    cy_edges = []
    cy_nodes = []
    print("START")
    for network_edge in edges:
        print(network_edge)
        try:
            source, target = network_edge.split(" ")
        except:
            break
        if source not in nodes:
            nodes.add(source)
            cy_nodes.append({"data": {"id": source, "label": "User #" + source[-5:]}})
        if target not in nodes:
            nodes.add(target)
            cy_nodes.append({"data": {"id": target, "label": "User #" + target[-5:]}})

        cy_edges.append({
            'data': {
                'source': source,
                'target': target
            }
        })


def get_tab_cytoscape_layout():
    styles = {
        'json-output': {
            'overflow-y': 'scroll',
            'height': 'calc(50% - 25px)',
            'border': 'thin lightgrey solid'
        },
        'tab': {
            'height': 'calc(98vh - 105px)'
        }
    }
    component = dcc.Tab(label='cytoscape', children=[html.Div([
        html.Div(className='four columns', children=[
            dcc.Tabs(id='tabs', children=[
                dcc.Tab(label='Control Panel', children=[
                    drc.NamedDropdown(
                        name='Layout',
                        id='dropdown-layout',
                        options=drc.DropdownOptionsList(
                            'random',
                            'grid',
                            'circle',
                            'concentric',
                            'breadthfirst',
                            'cose'
                        ),
                        value='grid',
                        clearable=False
                    ),

                    drc.NamedDropdown(
                        name='Node Shape',
                        id='dropdown-node-shape',
                        value='ellipse',
                        clearable=False,
                        options=drc.DropdownOptionsList(
                            'ellipse',
                            'triangle',
                            'rectangle',
                            'diamond',
                            'pentagon',
                            'hexagon',
                            'heptagon',
                            'octagon',
                            'star',
                            'polygon',
                        )
                    ),

                    drc.NamedInput(
                        name='Followers Color',
                        id='input-follower-color',
                        type='text',
                        value='#0074D9',
                    ),

                    drc.NamedInput(
                        name='Following Color',
                        id='input-following-color',
                        type='text',
                        value='#FF4136',
                    ),
                ]),

                dcc.Tab(label='JSON', children=[
                    html.Div(style=styles['tab'], children=[
                        html.P('Node Object JSON:'),
                        html.Pre(
                            id='tap-node-json-output',
                            style=styles['json-output']
                        ),
                        html.P('Edge Object JSON:'),
                        html.Pre(
                            id='tap-edge-json-output',
                            style=styles['json-output']
                        )
                    ])
                ])
            ]),
        ]),
    html.Div(className='eight columns', children=[
        cyto.Cytoscape(
            id='cytoscape',
            # elements=cy_edges + cy_nodes,
            style={
                'height': '95vh',
                'width': '100%'
            }
        )
    ]),
    ])])
    return component





if __name__ == '__main__':
    app.run_server(debug=True)