from maindash import app,type_storage, info_current_file
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

def make_layout():
    return html.Div(children=[
        html.Div(style={"padding-bottom": "10vh"},children=[
        # Barre de navigation
        html.Nav(style=nav_style, className="navbar", children=[
            html.Div(className="navbar-brand"),
            #dcc.Store(id='actual-tab-store', storage_type=type_storage),
            html.Div(className="navbar-menu", style={'justifyContent': 'center', 'width': '100%'}, children=[
                dcc.Tabs(id="tabs-example", value='tab-home', style={'width': '100%'},persistence=True,persistence_type=type_storage, children=[
                    dcc.Tab(label='Home', value='tab-home', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Data', value='tab-data', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Model', value='tab-model', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Visualization', value='tab-visualization', style=tab_style, selected_style=selected_tab_style),
                    dcc.Tab(label='Export results', value='tab-export', style=tab_style, selected_style=selected_tab_style)
                ])
            ])
        ]),

        # Contenu principal
        html.Div(id='tabs-content', className="container")
        ]),

        #html.Footer("This is the footer", style=footer_style)
    ])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-home':
        return layout_home()
    elif tab == 'tab-data':
        return layout_data()
    elif tab == 'tab-model':
        return layout_model()
    elif tab == 'tab-visualization':
        return layout_visualization()
    elif tab == 'tab-export':
        return layout_export_results()


if __name__=="__main__":
    #celery_app.start()
    app.layout=make_layout()
    app.run(host='127.0.0.1',port=8080,debug=True)