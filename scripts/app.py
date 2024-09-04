from maindash import app,type_storage
#from views.main_view import make_layout

import time
#import threading
import flask
import psutil
import os
import shutil
from dash import dcc, html
from dash.dependencies import Input, Output, State
# import plotly.express as px
# import os
# import re
# import base64
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
    "monitor_thread_launched_pid":False,
    "monitor_thread_launched_folder":False,
    "process_pid": None,
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
    #'status-run-model':'not-yet',
    'parameters_model':{
        'beta_matrix':{
            'apriori':'Horseshoe',
        },
        'precision_matrix':{
            'apriori':'Lasso',
            'parameters':{
                'lambda_init':10
            }
        },
        "nb_draws":1000,
        "nb_tune":2000,
        "target_accept":0.9,
    }
}


def make_layout():
    path_data="data/dash_app"
    if not os.path.exists(path_data):
        try:
            # Créer le dossier et ses parents si nécessaire
            os.makedirs(path_data)
            #print(f"Le dossier '{path_data}' a été créé avec succès.")
        except OSError as e:
            print(f"Erreur : Impossible de créer le dossier '{path_data}' : {e.strerror}")

    values_sliders={
        "credibility":0.95,
        "edges_width":5,
        "nodes_size":70,
        "font_size":20,
    }

    return html.Div(children=[
        dcc.Interval(id='check-deconnection',interval=5*1000,n_intervals=0),
        dcc.Store(id='status-sliders-co-occurence', storage_type=type_storage,data=values_sliders),
        dcc.Store(id='status-sliders-features-selection', storage_type=type_storage,data=values_sliders),
        dcc.Store(id='status-legend-covariates', storage_type=type_storage,data={"color":"#2acedd","text":"Covariates"}),
        html.Div(id="useless-component"),
        html.Div(style={"padding-bottom": "10vh"},children=[
        # Barre de navigation
        html.Nav(style=nav_style, className="navbar", children=[
            html.Div(className="navbar-brand"),
            dcc.Store(id='info-current-file-store', storage_type=type_storage,data=initial_info_current_file),
            dcc.Store(id='status-run-model', storage_type=type_storage,data="not-yet"),
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
        # html.Div(id='div-data',style={'display':'none'}),
        # html.Div(id='div-model',style={'display':'none'}),
        # html.Div(id='div-visu',style={'display':'none'}),
        #html.Div(id='div-export',style={'display':'none'},children=[layout_export_results()]),
        html.Div(id='div-export')
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
              Output('div-export', 'children'),
              Output('run-model-button','disabled',allow_duplicate=True),
              Output('run-model-button','title',allow_duplicate=True),
              #Input('tabs-mdine', 'value'),
              Input("actual-tab-store",'modified_timestamp'),
              State("actual-tab-store",'data'),
              State("info-current-file-store","data"),
              State("status-run-model","data"),prevent_initial_call=True)
def render_content(ts,tab,info_current_file_store,status_run_model):
    if ts is None:
        raise PreventUpdate
    display_none={'display':'none'}
    if tab == 'tab-home':
        return None,display_none,display_none,display_none,None,None,None
    elif tab == 'tab-data':
        return display_none,None,display_none,display_none,None,None,None
    elif tab == 'tab-model':
        if info_current_file_store["reference_taxa"]!=None and info_current_file_store["covar_end"]!=None and status_run_model=='not-yet' and info_current_file_store['phenotype_column']!='error':
            return display_none,display_none,None,display_none,None,False,None
        else:
            #At leat one error in the data section
            title="At least one error in the data section, please check the errors."
            return display_none,display_none,None,display_none,None,True,title
    elif tab == 'tab-visualization':
        return display_none,display_none,display_none,None,None,None,None
    elif tab == 'tab-export':
        return display_none,display_none,display_none,display_none,layout_export_results(status_run_model),None,None
    


@app.callback(Output('info-current-file-store','data',allow_duplicate=True),
              Input('check-deconnection','n_intervals'),
              State('info-current-file-store','data'),prevent_initial_call=True)
def register_connection(n_intervals,info_current_file_store):
    if info_current_file_store["session_folder"]!=None:
        txt_file=os.path.join(info_current_file_store["session_folder"],"time.txt")
        with open(txt_file, 'w') as f:
            f.write(f'{time.time()}\n')

        # if info_current_file_store["monitor_thread_launched_folder"]==False:
        #     print(f"Thread folder lauched {txt_file}")
        #     threading.Thread(target=monitor_disconnection,args=(info_current_file_store["session_folder"],info_current_file_store["process_pid"],"delete_folder",), daemon=True).start()
        #     info_current_file_store["monitor_thread_launched_folder"]=True

        # if info_current_file_store["monitor_thread_launched_pid"]==False and info_current_file_store["process_pid"]!=None and info_current_file_store["status-run-model"]=="in-progress":
        #     print(f"Thread pid lauched {txt_file}")
        #     threading.Thread(target=monitor_disconnection,args=(info_current_file_store["session_folder"],info_current_file_store["process_pid"],"kill_process",), daemon=True).start()
        #     info_current_file_store["monitor_thread_launched_pid"]=True
    
    return info_current_file_store

def monitor_disconnection(session_folder,pid_to_kill,type_action):
    txt_file=os.path.join(session_folder,"time.txt")
    while True:
        time.sleep(10)  # Vérifie toutes les 10 secondes
        try:
            with open(txt_file, 'r') as f:
                timestamps = f.readlines()
                if timestamps:
                    last_time=float(timestamps[-1].strip())
                    #print("Diff time:",time.time()-last_time)
        except FileNotFoundError:
            last_time=0
        if time.time() - last_time > 10:
            #print(f"L'utilisateur a probablement fermé la page {txt_file}")

            if type_action=="kill_process":
                #print("Je veux du kill")
                try:
                    process = psutil.Process(pid_to_kill)
                    process.terminate()  # Terminer le processus
                    #print(f"Processus avec PID {pid_to_kill} terminé avec succès.")
                except psutil.NoSuchProcess:
                    pass
                    #print(f"Processus avec PID {pid_to_kill} n'existe pas.")
                except psutil.AccessDenied:
                    pass
                    #print(f"Accès refusé pour terminer le processus avec PID {pid_to_kill}.")

            # Delete user folder
            elif type_action=="delete_folder":
                try:
                    # Suppression du dossier et de son contenu récursivement
                    shutil.rmtree(session_folder)
                    #print(f"Le dossier '{session_folder}' a été supprimé avec succès.")
                except OSError as e:
                    pass
                    #print(f"Erreur : {session_folder} : {e.strerror}")
            else:
                raise ValueError

            break

@app.server.route('/close-session', methods=['POST'])
def close_session():
    data = flask.request.json  # Data sent from JavaScript fetch
    #session_data = flask.request.environ.get('werkzeug.request').json['session-store']['data']
    #print(f"Session data: {data}")
    session_folder=data["session_folder"]
    pid_to_kill=data["process_pid"]
    if session_folder!=None:
        #print("Session folder: ",session_folder)
        txt_file=os.path.join(session_folder,"time.txt")
        time.sleep(30)
        with open(txt_file, 'r') as f:
            timestamps = f.readlines()
            if timestamps:
                last_time=float(timestamps[-1].strip())
                if time.time()-last_time>=20:
                    #Delete and stop everything related to the user session
                    #print(f"L'utilisateur a probablement fermé la page {txt_file}")
                    if pid_to_kill!=None:
                        try:
                            process = psutil.Process(pid_to_kill)
                            process.terminate()  # Terminer le processus
                            #print(f"Processus avec PID {pid_to_kill} terminé avec succès.")
                        except psutil.NoSuchProcess:
                            pass
                            #print(f"Processus avec PID {pid_to_kill} n'existe pas.")
                        except psutil.AccessDenied:
                            pass
                            #print(f"Accès refusé pour terminer le processus avec PID {pid_to_kill}.")

                    try:
                        # Suppression du dossier et de son contenu récursivement
                        shutil.rmtree(session_folder)
                        #print(f"Le dossier '{session_folder}' a été supprimé avec succès.")
                    except OSError as e:
                        pass
                        #print(f"Erreur : {session_folder} : {e.strerror}")
    
    return flask.jsonify({"status": "success"}), 200

if __name__=="__main__":
    app.layout=make_layout()
    app.run(host='0.0.0.0',port=8080,debug=True)