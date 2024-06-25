import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import os
import re
import base64
import zipfile
from io import BytesIO

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
    State("info-current-file-store","data"),
    prevent_initial_call=True
)
# def download_file(n_clicks):
#     # Chemin vers le fichier à télécharger
#     FILE_PATH = "data/dash_app/session_7/first_group/idata.pkl"
#     NEW_FILENAME = "nouveau_idata.pkl"  # Nom de fichier personnalisé
    

#     return dcc.send_file(FILE_PATH,filename=NEW_FILENAME)
def download_file(n_clicks,info_current_file_store):

    if info_current_file_store["status-run-model"]!="completed":
        raise PreventUpdate
    
    two_groups=info_current_file_store["second_group"]

    list_files=[]
   
    if two_groups!=None:
        #Two groups
        list_files.append(os.path.join(info_current_file_store["session_folder"],"first_group","idata.pkl"))
        list_files.append(os.path.join(info_current_file_store["session_folder"],"second_group","idata.pkl"))

    else:
        list_files.append(os.path.join(info_current_file_store["session_folder"],"idata.pkl"))

            

    # Nom du fichier ZIP
    zip_filename = "idata_raw_results.zip"

    # Créer un objet BytesIO pour stocker le fichier ZIP en mémoire
    zip_buffer = BytesIO()

    # Créer un fichier ZIP en mémoire
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in list_files:
            file_name = os.path.basename(file_path)
            zip_file.write(file_path, file_name)

    # Revenir au début du buffer pour l'envoyer
    zip_buffer.seek(0)

    return dcc.send_bytes(zip_buffer.getvalue(), filename=zip_filename)
