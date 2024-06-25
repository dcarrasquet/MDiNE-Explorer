from maindash import app,type_storage
#from views.main_view import make_layout

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import re
import base64
from dash.exceptions import PreventUpdate

from views.home_view import layout_home
from views.data_view import layout_data
from views.model_view import layout_model
from views.visualization_view import layout_visualization
from views.export_results_view import layout_export_results

#server = app.server

uploaded_file_name = None

# Styles pour la barre de navigation
nav_style = {
    'padding': '10px'
}

tab_style = {
    'backgroundColor': '#4A90E2',
    'color': 'white',
    'padding': '10px',
    'border': 'none',
    'textAlign': 'center',
    'width': '150px',  # Largeur élargie pour les onglets
    'borderRadius': '10px',
    'marginRight': '10px'
}

selected_tab_style = {
    'backgroundColor': '#341f97',
    'color': 'white',
    'padding': '10px',
    'border': 'none',
    'textAlign': 'center',
    'width': '150px',  # Largeur élargie pour les onglets sélectionnés
    'borderRadius': '10px',
    'marginRight': '10px'
}

input_field_style = {
    'width': '50%',
    'padding': '8px',
    'border': '1px solid #ccc',
    'borderRadius': '5px',
    'margin': '10px'
}

button_hover_style = {
    'background': '#4A90E2'
}

button_style = {
    'background': '#4CAF50',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'margin': '10px',
    'hover': button_hover_style
}

footer_style={
    'position': 'fixed',
    'bottom': 0,
    'width': '100%',
    'height': '10vh',
    'background-color': '#333',
    'color': 'white',
    'text-align': 'center',
    'padding': '20px 0',
}

initial_info_current_file={
    'filename':None,
    'session_folder':None,
    'nb_rows':None,
    'nb_columns':None,
    'covar_start':None,
    'covar_end':None,
    'taxa_start':None,
    'taxa_end':None,
    'reference_taxa':None,
    'phenotype_column':None,
    'first_group':None,
    'second_group':None,
    'filter_zeros':None,
    'filter_dev_mean':None,
    'status-run-model':'not-yet',
    'parameters_model':{
        'beta_matrix':{
            'apriori':'Ridge',
            'parameters':{
                'alpha':1,
                'beta':1
            }
        },
        'precision_matrix':{
            'apriori':'Lasso',
            'parameters':{
                'lambda_init':10
            }
        }
    }
}

def make_layout():
    return html.Div(children=[
        html.Div(style={"padding-bottom": "10vh"},children=[
        # Barre de navigation
        html.Nav(style=nav_style, className="navbar", children=[
            html.Div(className="navbar-brand"),
            dcc.Store(id='info-current-file-store', storage_type=type_storage,data=initial_info_current_file),
            dcc.Store(id='actual-tab-store', storage_type=type_storage,data="tab-home"),
            html.Div(className="navbar-menu", style={'justifyContent': 'center', 'width': '100%'}, children=[
                dcc.Tabs(id="tabs-mdine", value='tab-home', style={'width': '100%'},persistence=True,persistence_type=type_storage, children=[
                    dcc.Tab(label='Home', value='tab-home', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Data', value='tab-data', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Model', value='tab-model', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Visualization', value='tab-visualization', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Export results', value='tab-export', style=tab_style, selected_style=selected_tab_style)
                ])
            ])
        ]),

        # Contenu principal
        html.Div(className="container",children=[
            html.Div(id='div-home',children=[layout_home()]),
        html.Div(id='div-data',style={'display':'none'},children=[layout_data()]),
        html.Div(id='div-model',style={'display':'none'},children=[layout_model()]),
        html.Div(id='div-visu',style={'display':'none'},children=[layout_visualization()]),
        html.Div(id='div-export',style={'display':'none'},children=[layout_export_results()])
        ])
        
        ]),

        #html.Footer("This is the footer", style=footer_style)
    ])

@app.callback(Output("actual-tab-store",'data'),
              Input("tabs-mdine","value"))
def change_store_tab(tab):
    return tab


@app.callback(Output('div-home', 'style'),
              Output('div-data', 'style'),
              Output('div-model', 'style'),
              Output('div-visu', 'style'),
              Output('div-export', 'style'),
              Output('run-model-button','disabled',allow_duplicate=True),
              Output('run-model-button','title',allow_duplicate=True),
              #Input('tabs-mdine', 'value'),
              Input("actual-tab-store",'modified_timestamp'),
              State("actual-tab-store",'data'),

              State("info-current-file-store","data"),prevent_initial_call=True)
def render_content(ts,tab,info_current_file_store):
    if ts is None:
        raise PreventUpdate
    display_none={'display':'none'}
    #print("Tab:",tab)
    if tab == 'tab-home':
        return None,display_none,display_none,display_none,display_none,None,None
    elif tab == 'tab-data':
        return display_none,None,display_none,display_none,display_none,None,None
    elif tab == 'tab-model':
        if info_current_file_store["reference_taxa"]!=None and info_current_file_store["covar_end"]!=None and info_current_file_store["status-run-model"]=='not-yet':
            return display_none,display_none,None,display_none,display_none,False,None
        else:
            #At leat one error in the data section
            title="At least one error in the data section, please check the errors."
            return display_none,display_none,None,display_none,display_none,True,title
    elif tab == 'tab-visualization':
        return display_none,display_none,display_none,None,display_none,None,None
    elif tab == 'tab-export':
        return display_none,display_none,display_none,display_none,None,None,None


if __name__=="__main__":
    #celery_app.start()
    app.layout=make_layout()
    app.run(host='127.0.0.1',port=8080,debug=True)