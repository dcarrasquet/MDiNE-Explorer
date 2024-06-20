import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import re
import base64

from maindash import app

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
    'display': 'inline-block',
    'vertical-align': 'middle',
    'hover': button_hover_style
}

def layout_export_results():
    return html.Div([
            html.H3('Export Results Page'),
            html.Div([
                html.Button("Download raw results", id="download-button",style=button_style),
                dcc.Download(id="download")
            ])

        ])



# Callback pour gérer le téléchargement du fichier
@app.callback(
    Output("download", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_file(n_clicks):
    # Chemin vers le fichier à télécharger
    FILE_PATH = "data/dash_app/session_7/first_group/idata.pkl"
    NEW_FILENAME = "nouveau_idata.pkl"  # Nom de fichier personnalisé
    

    return dcc.send_file(FILE_PATH,filename=NEW_FILENAME)