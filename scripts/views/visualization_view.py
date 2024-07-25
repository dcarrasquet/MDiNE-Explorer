from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL
import pickle
import dash_cytoscape as cyto
cyto.load_extra_layouts()
from dash.exceptions import PreventUpdate
import json
import os
import uuid
import random
import re
#import dash_mantine_components as dmc

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from maindash import app, type_storage #info_current_file
from mdine.performance_metrics import get_energy_figure,get_acceptance_rate,get_trace_beta,get_trace_precision_matrix
from mdine.cytoscape_graph import get_elements_co_occurence_network,get_stylesheet_co_occurrence_network,get_stylesheet_diff_network
from mdine.extract_data_files import get_separate_data,get_df_taxa,get_list_taxa

button_download_style = {
    'background': '#4A90E2',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
    'margin-left': 'auto',
    'margin-right': '10%',
    #'margin-left': 'auto',
    #'margin-right': 'auto',
}

button_modify_legend_style= {
    'background': '#4A90E2',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
    # 'display': 'block',
    'margin-left': '10%',
    'margin-right': 'auto',
    #'margin-left': 'auto',
    #'margin-right': 'auto',
}

button_save_legend_style= {
    'background': '#4caf50',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
    # 'display': 'block',
    'margin-left': '10%',
    'margin-right': 'auto',
    #'margin-left': 'auto',
    #'margin-right': 'auto',
}

button_add_item_style= {
    'background': '#a51e4d',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
    # 'display': 'block',
    'margin-left': '10%',
    'margin-right': 'auto',
    #'margin-left': 'auto',
    #'margin-right': 'auto',
}

input_field_color_style = {
    'width': '10%',
    'padding': '8px',
    'border': '1px solid #ccc',
    'borderRadius': '5px',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
}

input_field_label_style = {
    'width': '10%',
    'padding': '8px',
    'border': '1px solid #ccc',
    'borderRadius': '5px',
    'margin': '10px',
    'display': 'inline-block',
    'vertical-align': 'middle',
}

style_dropdown=None
style_div_parent_dropdown={'display': 'flex', 'align-items': 'center'}
style_elements_associated={"margin":'30px'}
style_div_dropdown={'width': '50%'}

def layout_visualization():
    return html.Div([
            #html.H3('Visualization Page'),
            dcc.Store(id='legend-store', data=[],storage_type=type_storage),
            dcc.Store(id='tab-status', storage_type=type_storage),
            dcc.Tabs(id="subtabs", value='subtab-cooccurrence',style={'margin-bottom': '20px','display':'none'},persistence=True,persistence_type=type_storage, children=[
                dcc.Tab(label='Co-occurrence networks', value='subtab-cooccurrence'),
                #dcc.Tab(label='Features Selection', value='subtab-features-selection'),
                dcc.Tab(label='Performance metrics', value='subtab-performance')
            ]),
            html.Div(id='subtabs-content'),
        ])

@app.callback(Output('subtabs-content', 'children'),
              Output('subtabs', 'style'),
              [Input('subtabs', 'value'),State("legend-store",'data'),State('subtabs', 'style'),Input("info-current-file-store",'data')])
def render_subcontent(subtab,legend_store,subtabs_style,info_current_file_store):

    ctx = callback_context

    if not ctx.triggered:
        trigger = 'No input has triggered yet'
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger=="info-current-file-store":
        # if subtabs_style!={'display':'none'}{'margin-bottom': '20px','display':'none'}:
        if subtabs_style!={'margin-bottom': '20px','display':'none'}:
            raise PreventUpdate
    
    if info_current_file_store["status-run-model"]=='not-yet':
        return "The inference has not been launched. Press Run model in the Model tab to view results.",{'margin-bottom': '20px','display':'none'}
    elif info_current_file_store["status-run-model"]=="in-progress":
        return "The model is running. You can view the current status in the Model tab. Come back later to see the results",{'margin-bottom': '20px','display':'none'}
    elif info_current_file_store["status-run-model"]=="error":
        return "An error occurred during model execution. Close the current tab and start a new simulation.",{'margin-bottom': '20px','display':'none'}
    elif info_current_file_store["status-run-model"]=="completed":
        if subtab == 'subtab-cooccurrence':
            return layout_co_occurence_networks(legend_store,info_current_file_store),{'margin-bottom': '20px'}
        elif subtab == 'subtab-performance':
            return layout_performance_metrics(),{'margin-bottom': '20px'}
        elif subtab== 'subtab-features-selection':
            return layout_features_selection(),{'margin-bottom': '20px'}
    else:
        print("info current file status run odel not correct")
        return None,{'margin-bottom': '20px','display':'none'}
        

def layout_co_occurence_networks(legend_store,info_current_file_store):

    switch_networks_diff=None

    if info_current_file_store["phenotype_column"]==None:
        switch_networks_diff=html.Div()
        file_idata=os.path.join(info_current_file_store["session_folder"],"idata.pkl")
        df_taxa=get_df_taxa(info_current_file_store,"df_taxa")
    else:
        switch_networks_diff=dcc.RadioItems(
            id='checklist-networks',
            options=[
                {'label': 'First group co-occurrence network', 'value': 'first_group'},
                {'label': 'Second group co-occurrence network', 'value': 'second_group'},
                {'label': 'Differential co-occurrence network', 'value': 'diff_network'},
            ],
            value="first_group",
            persistence=True,
            persistence_type=type_storage,
            style={'margin': '20px 0'}
        )
        file_idata=os.path.join(info_current_file_store["session_folder"],"first_group","idata.pkl")
        [df_covariates_1,df_taxa_1],[df_covariates_2,df_taxa_2]=get_separate_data(info_current_file_store)
        df_taxa=df_taxa_1
        

    

    with open(file_idata, "rb") as f:
        idata = pickle.load(f)

    
    elements=get_elements_co_occurence_network(df_taxa,legend_store)
    stylesheet=get_stylesheet_co_occurrence_network(idata,df_taxa,legend_store,hdi=0.95,node_size=0.01,edge_width=5,font_size=20)

    return html.Div([
                html.Div(id="jesersarien"),
                html.Div(id="state-separate-groups",children=[switch_networks_diff]),
                cyto.Cytoscape(
                    id='cytoscape',
                    elements=elements,
                    stylesheet=stylesheet,
                    autoRefreshLayout=True,
                    style={'width': '90%','height':'50vh','backgroundColor': '#f0f0f0','margin': '0 auto'},
                    #layout={'name': 'concentric'} #cose #concentric
                    layout={'name': 'preset'}
                    
                ),
                
                html.Div([
                    dcc.Location(id='refresh-url', refresh=True),
                    html.Button("Modify legend", id="button-modify-legend",style=button_modify_legend_style),
                    html.Button("Download graph", id="button-download-cytoscape",style=button_download_style),

                ],style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'}),
                
                html.Div(id="div-sliders-graph",children=[

                html.Div([
            html.Label('Credibility'),
            dcc.Slider(
                id='slider-credibility',
                min=0.01,
                max=0.99,
                #step=1,
                value=0.95,
                #marks={i: str(i) for i in range(0, 101, 10)}
                persistence=True,
                persistence_type=type_storage,
            ),
        ], style={'width': '90%','margin': '0 auto'}),

        # Slider 2
        html.Div([
            html.Label('Edges Width'),
            dcc.Slider(
                id='slider-edges-width',
                min=0,
                max=10,
                #step=1,
                value=1,
                #marks={i: str(i) for i in range(0, 101, 10)}
                persistence=True,
                persistence_type=type_storage,
            ),
        ], style={'width': '90%','margin': '0 auto'}),

        # Slider 3
        html.Div([
            html.Label('Nodes Size'),
            dcc.Slider(
                id='slider-nodes-size',
                min=0.001,
                max=1,
                #step=1,
                value=0.01,
                #marks={i: str(i) for i in range(0, 101, 10)}
                persistence=True,
                persistence_type=type_storage,
            ),
        ], style={'width': '90%','margin': '0 auto'}),

        # Slider 4
        html.Div([
            html.Label('Font Size'),
            dcc.Slider(
                id='slider-font-size',
                min=1,
                max=100,
                #step=1,
                value=20,
                #marks={i: str(i) for i in range(0, 101, 10)}
                persistence=True,
                persistence_type=type_storage,
            ),
        ], style={'width': '90%','margin': '0 auto'}),
        ]),

        html.Div(id="div-modify-legend",style={'display':'none'},children=[
            #dcc.Store(id='legend-store', data=[],storage_type=type_storage),
            #html.H5("Coucou"),
            #html.Div(id='selected-nodes-output'),
            # html.Div(id='container-items-add', children=[
            # html.Div(id='items-container'),
            # html.Button('Add Item', id='add-item', n_clicks=0,style=button_add_item_style)
            # ]),
            html.Div(id='items-container'),
            html.Button('Add a group to the legend', id='add-item', n_clicks=0,style=button_add_item_style),
            #html.Div(id='list-items-selection'),
            
            
            
        ])
                

           ])

# @app.callback(
#     Output('jesersarien', 'children'),
#     Input('cytoscape', 'stylesheet'),
#     Input('cytoscape', 'elements'))
# def test(stylesheet,elements):
#     print("Stylesheet: ",stylesheet)
#     print("Elements: ",elements)
#     return []

# Callback pour mettre à jour le graphique Cytoscape
@app.callback(
    Output('cytoscape', 'stylesheet',allow_duplicate=True),
    Input('slider-credibility', 'value'),
    Input('slider-edges-width', 'value'),
    Input('slider-nodes-size', 'value'),
    Input('slider-font-size', 'value'),
    State('state-separate-groups','children'),
    State('legend-store','data'),
    State("info-current-file-store","data"),prevent_initial_call=True
)
def update_graph(credibility, edges_width, nodes_size,font_size,children,legend_store,info_current_file_store):

    specific_graph=None


    try:
        specific_graph=children[0]['props']['value']
    except:
        specific_graph=None

    if specific_graph==None:
        file_idata=os.path.join(info_current_file_store["session_folder"],"idata.pkl")

        df_taxa=get_df_taxa(info_current_file_store,"df_taxa")

        with open(file_idata, "rb") as f:
            idata = pickle.load(f)

        stylesheet=get_stylesheet_co_occurrence_network(idata,df_taxa,legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
    else:
        separate_data=get_separate_data(info_current_file_store)
        df_taxa1=separate_data[0][1]
        df_taxa2=separate_data[1][1]

        # file_idata1=os.path.join(info_current_file["session_folder"],"first_group","idata.pkl")
        # file_idata2=os.path.join(info_current_file["session_folder"],"second_group","idata.pkl")

        file_idata1=os.path.join(info_current_file_store["session_folder"],"first_group","idata.pkl")
        file_idata2=os.path.join(info_current_file_store["session_folder"],"second_group","idata.pkl")


        # df_taxa=get_df_file(info_current_file).iloc[:,5:11]

        with open(file_idata1, "rb") as f1:
            idata1 = pickle.load(f1)

        with open(file_idata2, "rb") as f2:
            idata2 = pickle.load(f2)

        if specific_graph=='first_group':
            stylesheet=get_stylesheet_co_occurrence_network(idata1,df_taxa1,legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
        elif specific_graph=='second_group':
            stylesheet=get_stylesheet_co_occurrence_network(idata2,df_taxa2,legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
        elif specific_graph=='diff_network':
            stylesheet=get_stylesheet_diff_network(idata1,df_taxa1,idata2,df_taxa2,legend_store=legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
        
    return stylesheet

@app.callback(
    Output("cytoscape", "generateImage"),
    Input("button-download-cytoscape", "n_clicks"))
def get_image(n_clicks):

    if n_clicks==None:
        raise PreventUpdate
    
    action = "download"
    ftype = "svg"
    filename = "my_cytoscape_graph"  # Nom du fichier

    return {
        'type': ftype,
        'action': action,
        'filename': filename
        }


@app.callback(
    Output('cytoscape', 'stylesheet',allow_duplicate=True),
    Input('checklist-networks', 'value'),
    State('slider-credibility', 'value'),
    State('slider-edges-width', 'value'),
    State('slider-nodes-size', 'value'),
    State('slider-font-size', 'value'),
    State('legend-store','data'),
    State("info-current-file-store","data"),prevent_initial_call=True
)
def change_stylesheet(value,credibility, edges_width, nodes_size,font_size,legend_store,info_current_file_store):
    ## Change between groups if the user separates the data
    ## Function called only if there are two different groups


    separate_data=get_separate_data(info_current_file_store)
    df_taxa1=separate_data[0][1]
    df_taxa2=separate_data[1][1]

    file_idata1=os.path.join(info_current_file_store["session_folder"],"first_group","idata.pkl")
    file_idata2=os.path.join(info_current_file_store["session_folder"],"second_group","idata.pkl")

    with open(file_idata1, "rb") as f1:
        idata1 = pickle.load(f1)

    with open(file_idata2, "rb") as f2:
        idata2 = pickle.load(f2)
    stylesheet=None

    if value=='first_group':
        stylesheet=get_stylesheet_co_occurrence_network(idata1,df_taxa1,legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
    elif value=='second_group':
        stylesheet=get_stylesheet_co_occurrence_network(idata2,df_taxa2,legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
    elif value=='diff_network':
        stylesheet=get_stylesheet_diff_network(idata1,df_taxa1,idata2,df_taxa2,legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
        
    return stylesheet


def layout_performance_metrics():
    return html.Div([
            html.H4('Performance metrics'),
            html.Div([
    html.Label("Select the figures you want to plot:", style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='dropdown-performances-metrics',
        options=[
            {'label': 'Energy Graph', 'value': 'energy'},
            {'label': 'Acceptance Rate', 'value': 'acceptance_rate'},
            {'label': 'Beta Matrix Trace', 'value': 'beta_matrix_trace'},
            {'label': 'Precision Matrix Trace', 'value': 'precision_matrix_trace'}
        ],
        placeholder="Select figures",
        multi=True,
        style={'width': '50%', 'margin': '20px 0'},
        persistence=True,
        persistence_type=type_storage,
    ),
    html.Div(id="output-graph-dropdown",children=[]),
    #dcc.Graph(figure=trace_beta,style={'width': '90%','margin': '0 auto'}),

    ])
        ])

@app.callback(
    Output("output-graph-dropdown","children"),
    Input("dropdown-performances-metrics","value"),
    State("info-current-file-store","data"),
    #State("output-graph-dropdown"),
)
def output_dropdown(list_graphs,info_current_file_store):

    new_children=[]

    two_groups=info_current_file_store["second_group"]


    if list_graphs!=None:

        if two_groups!=None:
            #Two groups
            file_idata1=os.path.join(info_current_file_store["session_folder"],"first_group","idata.pkl")
            file_idata2=os.path.join(info_current_file_store["session_folder"],"second_group","idata.pkl")

            with open(file_idata1, "rb") as f:
                idata1 = pickle.load(f)

            with open(file_idata2, "rb") as f:
                idata2 = pickle.load(f)

        else:
            file_idata=os.path.join(info_current_file_store["session_folder"],"idata.pkl")

            with open(file_idata, "rb") as f:
                idata = pickle.load(f)

        for graph in list_graphs:

            if graph=="energy":
                if two_groups==None:
                    energy_figure=get_energy_figure(idata)

                    new_children.append(html.H5(
                        "Energy Graph",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))
                    
                    new_children.append(dcc.Graph(figure=energy_figure,style={'width': '90%','margin': '0 auto'}))
                else:
                    energy_figure1=get_energy_figure(idata1)

                    new_children.append(html.H5(
                        "Energy Graph (First Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))
                    
                    new_children.append(dcc.Graph(figure=energy_figure1,style={'width': '90%','margin': '0 auto'}))

                    energy_figure2=get_energy_figure(idata2)

                    new_children.append(html.H5(
                        "Energy Graph (Second Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))
                    
                    new_children.append(dcc.Graph(figure=energy_figure2,style={'width': '90%','margin': '0 auto'}))

            elif graph=="acceptance_rate":
                if two_groups==None:
                    acceptance_rate=get_acceptance_rate(idata)

                    new_children.append(html.H5(
                        "Acceptance Rate",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))

                    new_children.append(dcc.Graph(figure=acceptance_rate,style={'width': '90%','margin': '0 auto'}))
                else:
                    acceptance_rate1=get_acceptance_rate(idata1)

                    new_children.append(html.H5(
                        "Acceptance Rate (First Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))

                    new_children.append(dcc.Graph(figure=acceptance_rate1,style={'width': '90%','margin': '0 auto'}))

                    acceptance_rate2=get_acceptance_rate(idata2)

                    new_children.append(html.H5(
                        "Acceptance Rate (Second Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))

                    new_children.append(dcc.Graph(figure=acceptance_rate2,style={'width': '90%','margin': '0 auto'}))


            elif graph=="beta_matrix_trace":

                if two_groups==None:

                    trace_beta1,trace_beta2=get_trace_beta(idata)

                    element=html.Div([
                        html.Div([
                            dcc.Graph(figure=trace_beta1),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(figure=trace_beta2),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ])

                    new_children.append(html.H5(
                        "Beta Matrix Graph",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))
                
                    new_children.append(element)

                else:

                    trace_beta1_group1,trace_beta2_group1=get_trace_beta(idata1)

                    element=html.Div([
                        html.Div([
                            dcc.Graph(figure=trace_beta1_group1),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(figure=trace_beta2_group1),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ])

                    new_children.append(html.H5(
                        "Beta Matrix Graph (First Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))
                
                    new_children.append(element)

                    trace_beta1_group2,trace_beta2_group2=get_trace_beta(idata2)

                    element=html.Div([
                        html.Div([
                            dcc.Graph(figure=trace_beta1_group2),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(figure=trace_beta2_group2),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ])

                    new_children.append(html.H5(
                        "Beta Matrix Graph (Second Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))
                
                    new_children.append(element)


            elif graph=="precision_matrix_trace":
                if two_groups==None:
                    trace_precision1,trace_precision2=get_trace_precision_matrix(idata)

                    element=html.Div([
                        html.Div([
                            dcc.Graph(figure=trace_precision1),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(figure=trace_precision2),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ])

                    new_children.append(html.H5(
                        "Precision Matrix Trace",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))

                    new_children.append(element)

                else:

                    trace_precision1_group1,trace_precision2_group1=get_trace_precision_matrix(idata1)

                    element=html.Div([
                        html.Div([
                            dcc.Graph(figure=trace_precision1_group1),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(figure=trace_precision2_group1),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ])

                    new_children.append(html.H5(
                        "Precision Matrix Trace (First Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))

                    new_children.append(element)

                    trace_precision1_group2,trace_precision2_group2=get_trace_precision_matrix(idata2)

                    element=html.Div([
                        html.Div([
                            dcc.Graph(figure=trace_precision1_group2),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(figure=trace_precision2_group2),
                        ], style={'width': '50%', 'display': 'inline-block'}),
                    ])

                    new_children.append(html.H5(
                        "Precision Matrix Trace (Second Group)",
                        style={'textAlign': 'center', 'fontWeight': 'bold'}
                    ))

                    new_children.append(element)

            else:
                print("Error invalid graph name")
        
    return new_children

# Callback pour capturer la sélection des nœuds
#div-download-sliders-graph
#div-modify-legend
@app.callback(
    #Output('selected-nodes-output', 'children'),
    Output('div-sliders-graph', 'style'),
    Output('div-modify-legend', 'style'),
    Output("button-modify-legend","children"),
    Output("button-modify-legend","style"),
    Output("items-container","children"),
    #Output('refresh-url', 'href'),
    Output("cytoscape","elements"),
    Output("cytoscape","stylesheet"),
    Input("button-modify-legend","n_clicks"),
    State('legend-store','data'),
    State("info-current-file-store","data"),
    State("cytoscape","elements"),
    State('state-separate-groups','children'),
    State("cytoscape","stylesheet"),
    State('slider-credibility', 'value'),
    State('slider-edges-width', 'value'),
    State('slider-nodes-size', 'value'),
    State('slider-font-size', 'value'),
)
def display_selected_nodes(n_clicks,data_legend_store,info_current_file_store,cytoscape_elements,state_separate_groups,current_stylesheet,credibility, edges_width, nodes_size,font_size):


    if n_clicks==None:
        raise PreventUpdate
    elif n_clicks%2==0:
        # Return to sliders
        df_taxa=get_df_taxa(info_current_file_store,"df_taxa")
        elements=get_elements_co_occurence_network(df_taxa,data_legend_store)

        specific_graph=None


        try:
            specific_graph=state_separate_groups[0]['props']['value']
        except:
            specific_graph=None

        if specific_graph==None:
            file_idata=os.path.join(info_current_file_store["session_folder"],"idata.pkl")

            df_taxa=get_df_taxa(info_current_file_store,"df_taxa")

            with open(file_idata, "rb") as f:
                idata = pickle.load(f)

            stylesheet=get_stylesheet_co_occurrence_network(idata,df_taxa,data_legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
        else:
            separate_data=get_separate_data(info_current_file_store)
            df_taxa1=separate_data[0][1]
            df_taxa2=separate_data[1][1]

            # file_idata1=os.path.join(info_current_file["session_folder"],"first_group","idata.pkl")
            # file_idata2=os.path.join(info_current_file["session_folder"],"second_group","idata.pkl")

            file_idata1=os.path.join(info_current_file_store["session_folder"],"first_group","idata.pkl")
            file_idata2=os.path.join(info_current_file_store["session_folder"],"second_group","idata.pkl")


            # df_taxa=get_df_file(info_current_file).iloc[:,5:11]

            with open(file_idata1, "rb") as f1:
                idata1 = pickle.load(f1)

            with open(file_idata2, "rb") as f2:
                idata2 = pickle.load(f2)

            if specific_graph=='first_group':
                stylesheet=get_stylesheet_co_occurrence_network(idata1,df_taxa1,data_legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
            elif specific_graph=='second_group':
                stylesheet=get_stylesheet_co_occurrence_network(idata2,df_taxa2,data_legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)
            elif specific_graph=='diff_network':
                stylesheet=get_stylesheet_diff_network(idata1,df_taxa1,idata2,df_taxa2,legend_store=data_legend_store,hdi=credibility,node_size=nodes_size,edge_width=edges_width,font_size=font_size)


        return None,{'display':'none'},"Modify legend",button_modify_legend_style,[],elements,stylesheet
    else:
        # Become Save Legend
        # Display Legend attributes
        remaining_taxa=get_list_remaining_taxa(data_legend_store,info_current_file_store)
        list_items=create_items(data_legend_store,remaining_taxa)

            

        return {'display':'none'},None,"Save legend",button_save_legend_style,list_items,cytoscape_elements,current_stylesheet
    



@app.callback(
    Output('legend-store', 'data',allow_duplicate=True),
    Output('items-container', 'children',allow_duplicate=True),
    Input('add-item', 'n_clicks'),
    Input({'type': 'trash-icon', 'index': ALL}, 'n_clicks'),
    State('legend-store', 'data'),
    State('items-container', 'children'),
    State("info-current-file-store","data"),prevent_initial_call=True
)
def display_items(add_clicks, trash_clicks,legend,children,info_current_file_store):

    if children is None:
        children = []

    nb_children=len(children)

    ctx = callback_context

    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if 'add-item' in triggered_id and add_clicks!=0:
        new_color=random_hex_color()
        new_element_id = str(uuid.uuid4())
        new_group = {
        'id': new_element_id,
        'color': new_color,
        'label': f'Group {nb_children + 1}',
        'elements': []
        
        }
        legend.append(new_group)

    else:
        # difference between initial call and true call of trash icon
        trash=False
        for clicks in trash_clicks:
            if clicks!=None:
                trash=True
        if trash==True:
            trash_index = eval(triggered_id)['index']
            legend=[group for group in legend if group['id']!=trash_index]
    
    remaining_taxa=get_list_remaining_taxa(legend,info_current_file_store)

    list_items=create_items(legend,remaining_taxa)

    return legend,list_items

def random_hex_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


@app.callback(
    Output('legend-store', 'data',allow_duplicate=True),
    Output({'type': 'color-box', 'index': ALL}, 'style'),
    Input({'type': 'color-input', 'index': ALL}, 'value'),
    Input({'type': 'label-input', 'index': ALL}, 'value'),
    State('legend-store', 'data'),
    prevent_initial_call=True
)
def update_group_data(new_color_list,new_label_list, store_data):

    

    ctx = callback_context

    if [trigger['prop_id'].split('.')[0] for trigger in ctx.triggered]==['']:
        raise PreventUpdate

    triggered_inputs = [json.loads(trigger['prop_id'].split('.')[0]) for trigger in ctx.triggered]

    

    list_id=remove_duplicates([triggered_input['index'] for triggered_input in triggered_inputs])

    if len(new_color_list)==len(store_data):
        #print("All components have triggered the callback")
        ## All components have triggered the callback
        for idx,group in enumerate(store_data):
            group['color'] = new_color_list[idx] 
            group['label'] = new_label_list[idx]
    else:
        ##Only one component has triggered
        triggered_id=triggered_inputs[0]
        for group in store_data:
            if group['id']==triggered_id:
                #idx=list_id.index(triggered_id)
                group['color'] = new_color_list[0] 
                group['label'] = new_label_list[0]
                break

    list_color_box=[]
    for color_value in new_color_list:
        if is_valid_hex_color(color_value):
            list_color_box.append({'width': '20px', 'height': '20px', 'backgroundColor': color_value,'margin-left':"1%"})
        else:
            list_color_box.append({'width': '20px', 'height': '20px', 'backgroundColor': "#FFFFFF",'margin-left':"1%"})


    return store_data,list_color_box

@app.callback(
    Output('legend-store', 'data',allow_duplicate=True),
    Output({'type': 'dropdown-group-items', 'index': ALL}, 'options'),
    Input({'type': 'dropdown-group-items', 'index': ALL}, 'value'),
    State('legend-store', 'data'),
    State('info-current-file-store', 'data'),
    prevent_initial_call=True
)
def update_group_data(list_value_dropdown, store_data,info_current_file_store):

    #print("Callback dropdown group items lauched")
    #print('jenairammre:',list_value_dropdown)
    ctx = callback_context

    if [trigger['prop_id'].split('.')[0] for trigger in ctx.triggered]==['']:
        raise PreventUpdate

    triggered_inputs = [json.loads(trigger['prop_id'].split('.')[0]) for trigger in ctx.triggered]
    list_id=remove_duplicates([triggered_input['index'] for triggered_input in triggered_inputs])


    if len(list_value_dropdown)==len(store_data):
        ## All components have triggered the callback
        for idx,group in enumerate(store_data):
            group['elements'] = list_value_dropdown[idx]
            
    else:
        ##Only one component has triggered
        triggered_id=triggered_inputs[0]
        for group in store_data:
            if group['id']==triggered_id:
                #idx=list_id.index(triggered_id)
                group['elements'] = list_value_dropdown[0] 
                
    remaining_taxa=get_list_remaining_taxa(store_data,info_current_file_store)

    options_dropdown=[]
    for taxa in remaining_taxa:
        options_dropdown.append({'label': taxa, 'value': taxa})

    return_list_dropdown_options=[]

    #print('Options dropdown: ',options_dropdown)

    for value_dropdown in list_value_dropdown:
        others_options=[]
        for element in value_dropdown:
            others_options.append({'label': element, 'value': element})
        #print("Others options: ",others_options)
        return_list_dropdown_options.append(options_dropdown+others_options)

    #print("Return list dropdown: ",return_list_dropdown_options)


    return store_data,return_list_dropdown_options


def is_valid_hex_color(hex_color):
    return re.fullmatch(r'^#(?:[0-9a-fA-F]{3}){1,2}$', hex_color) is not None

def remove_duplicates(original_list):
    seen = set()
    unique_list = []
    for item in original_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list

def get_list_remaining_taxa(data_legend_store,info_current_file_store):
    #print('Data legend store: ',data_legend_store)
    list_remaining_taxa=get_list_taxa(info_current_file_store)[:-1]
    #print("Toutes les taxa du début: ",list_remaining_taxa)
    if data_legend_store!=[]:
        for group in data_legend_store:
            if len(group["elements"])!=0:
                for element in group["elements"]:
                    list_remaining_taxa.remove(element)
                        

    return list_remaining_taxa
                    
def print_legend_store(legend_store):
    for group in legend_store:
        print(f"{group["label"]}, color: {group["color"]}, elements: {group["elements"]}")

def create_items(legend_store,remaining_taxa):
    options_dropdown=[]
    list_items=[]
    for taxa in remaining_taxa:
        options_dropdown.append({'label': taxa, 'value': taxa})
    
    for group in legend_store:
        options_dropdown=[]
        for taxa in remaining_taxa:
            options_dropdown.append({'label': taxa, 'value': taxa})
        for element in group["elements"]:
            options_dropdown.append({'label': element, 'value': element})

        color_value=group["color"] if is_valid_hex_color(group["color"]) else '#FFFFFF'

        element = html.Div([
                        html.Div(
                            id={'type': 'color-box', 'index': group["id"]},
                            style={'width': '20px', 'height': '20px', 'backgroundColor': color_value, 'margin-left': "1%"}
                        ),
                        dcc.Input(
                            type='text',
                            placeholder='Enter hex color value',
                            value=group["color"],
                            style={'flex': '0 1 10%', 'margin-right': '1%','width': '10%','padding': '8px','border': '1px solid #ccc','borderRadius': '5px','margin': '10px','vertical-align': 'middle'},
                            id={'type': 'color-input', 'index': group["id"]}
                        ),
                        dcc.Input(
                            type='text',
                            placeholder='Enter label',
                            id={'type': 'label-input', 'index': group["id"]},
                            value=group["label"],
                            style={'flex': '0 1 15%', 'margin-right': '1%','width': '10%','padding': '8px','border': '1px solid #ccc','borderRadius': '5px','margin': '10px','vertical-align': 'middle',}
                        ),
                        html.Div(
                            style={'flex': '1', 'display': 'flex', 'align-items': 'center'},
                            children=[
                                html.H5(
                                    "Elements associated: ",
                                    style={"margin": '0 10px 0 0'}
                                ),
                                dcc.Dropdown(
                                    id={'type': 'dropdown-group-items', 'index': group["id"]},
                                    value=group["elements"],
                                    options=options_dropdown,
                                    placeholder="Select elements",
                                    multi=True,
                                    style={'flex': '1','margin-right':"3%"},
                                    persistence=True,
                                    persistence_type=type_storage,
                                )
                            ]
                        ),
                        html.I(
                            className="fas fa-trash",
                            id={'type': 'trash-icon', 'index': group["id"]},
                            style={'color': 'red', 'cursor': 'pointer', 'margin': '5px', 'margin-left': 'auto', 'margin-right': '3%'}
                        ),
                    ], id=group["id"], style={'width': '90%', 'margin': '0 auto', 'border': '1px solid black', 'border-radius': '10px', 'padding': '10px', 'display': 'flex', 'align-items': 'center', 'margin-top': '5px'})
        list_items.append(element)

    return list_items


def layout_features_selection():
    return html.H5("Features Selection view")